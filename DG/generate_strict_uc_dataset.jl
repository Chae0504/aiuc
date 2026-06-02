using JuMP
using Gurobi
using CSV
using DataFrames
using Distributions
using Random
using NPZ
using Serialization

const MOI = JuMP.MOI

const NUM_HOURS = 24

function env_int(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float(name::String, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_path(name::String, default::String)
    value = get(ENV, name, default)
    return isabspath(value) ? value : joinpath(@__DIR__, value)
end

function log_message(message)
    println(message)
    flush(stdout)
end

function load_specs(path::String)
    specs = CSV.read(path, DataFrame)
    required = [
        :Generator,
        :MaxProd,
        :MinProd,
        :IniProd,
        :IniState,
        :SUcap,
        :SDcap,
        :RampUp,
        :RampDw,
        :SlopeVarCost,
        :InterVarCost,
        :SUcost1,
        :ShutdownCost,
        :MinTU,
        :MinTD,
    ]
    missing_columns = setdiff(required, propertynames(specs))
    isempty(missing_columns) || error("Missing columns in $(path): $(missing_columns)")

    numeric_columns = setdiff(required, [:Generator])
    for column in numeric_columns
        any(ismissing, specs[!, column]) && error("Missing value in column $(column)")
        specs[!, column] = Float64.(specs[!, column])
    end

    specs[!, :InitStatus] = Int.(specs.IniState .> 0.0)
    specs[!, :PastHours] = abs.(specs.IniState)

    all(specs.MinProd .>= 0.0) || error("MinProd must be non-negative")
    all(specs.MaxProd .>= specs.MinProd) || error("MaxProd must be at least MinProd")
    all((specs.SUcap .>= specs.MinProd) .& (specs.SUcap .<= specs.MaxProd)) ||
        error("SUcap must be between MinProd and MaxProd")
    all((specs.SDcap .>= specs.MinProd) .& (specs.SDcap .<= specs.MaxProd)) ||
        error("SDcap must be between MinProd and MaxProd")
    all((specs.RampUp .>= 0.0) .& (specs.RampDw .>= 0.0)) ||
        error("Ramp limits must be non-negative")
    all(x -> isinteger(x) && x >= 0.0, specs.MinTU) ||
        error("MinTU must contain non-negative integers")
    all(x -> isinteger(x) && x >= 0.0, specs.MinTD) ||
        error("MinTD must contain non-negative integers")

    for g in 1:nrow(specs)
        if specs.InitStatus[g] == 1
            specs.MinProd[g] <= specs.IniProd[g] <= specs.MaxProd[g] ||
                error("Initial output for $(specs.Generator[g]) is outside its online range")
        else
            abs(specs.IniProd[g]) <= 1e-6 ||
                error("Offline generator $(specs.Generator[g]) has non-zero initial output")
        end
    end

    return specs
end

function base_demand_profile()
    return [
        33144.0, 30107.2, 24033.6, 14368.0, 17960.0, 25552.0,
        33144.0, 39217.6, 42254.4, 46809.6, 47568.8, 43772.8,
        40736.0, 37699.2, 46809.6, 48328.0, 44532.0, 47568.8,
        51364.8, 54401.6, 55920.0, 48328.0, 46050.4, 42254.4,
    ] ./ 10.0
end

function sample_demand(
    rng::AbstractRNG,
    base_demand::Vector{Float64},
    accepted_index::Int,
    target_samples::Int,
)
    daily_scale = rand(rng, Uniform(0.7, 1.2))
    phase_shift = rand(rng, Uniform(0.0, 2pi))
    wave_amplitude = rand(rng, Uniform(-0.15, 0.15))
    wave = 1.0 .+ wave_amplitude .* sin.(range(0.0, 2pi, length=NUM_HOURS) .+ phase_shift)
    morphed_demand = base_demand .* daily_scale .* wave

    noise_distribution =
        accepted_index <= target_samples ÷ 2 ? Normal(0.0, 0.05) : Uniform(-0.08, 0.08)
    noisy_demand = max.(morphed_demand .* (1.0 .+ rand(rng, noise_distribution, NUM_HOURS)), 0.0)

    smoothed_demand = similar(noisy_demand)
    for t in 1:NUM_HOURS
        previous_t = t == 1 ? NUM_HOURS : t - 1
        next_t = t == NUM_HOURS ? 1 : t + 1
        smoothed_demand[t] = (noisy_demand[previous_t] + noisy_demand[t] + noisy_demand[next_t]) / 3.0
    end
    return smoothed_demand
end

function write_dataset(
    path::String,
    x_demand,
    y_power,
    y_status,
    daily_cost,
    accepted::Int,
    attempts::Int,
    target_samples::Int,
    random_seed::Int,
    mip_gap::Float64,
)
    NPZ.npzwrite(
        path,
        Dict(
            "X_demand" => x_demand[1:accepted, :, :],
            "Y_power" => y_power[1:accepted, :, :],
            "Y_status" => y_status[1:accepted, :, :],
            "daily_cost" => daily_cost[1:accepted],
            "accepted_samples" => [accepted],
            "generation_attempts" => [attempts],
            "target_samples" => [target_samples],
            "random_seed" => [random_seed],
            "mip_gap" => [mip_gap],
        ),
    )
end

function save_checkpoint!(
    output_path::String,
    x_demand,
    y_power,
    y_status,
    daily_cost,
    accepted::Int,
    attempts::Int,
    target_samples::Int,
    random_seed::Int,
    mip_gap::Float64,
    rng::AbstractRNG,
)
    checkpoint_path = output_path * ".partial.npz"
    rng_path = output_path * ".partial.rng"
    next_checkpoint_path = checkpoint_path * ".next"
    next_rng_path = rng_path * ".next"

    write_dataset(
        next_checkpoint_path,
        x_demand,
        y_power,
        y_status,
        daily_cost,
        accepted,
        attempts,
        target_samples,
        random_seed,
        mip_gap,
    )
    open(next_rng_path, "w") do io
        serialize(io, rng)
    end
    mv(next_checkpoint_path, checkpoint_path; force=true)
    mv(next_rng_path, rng_path; force=true)
    log_message("Checkpoint saved at $(accepted)/$(target_samples) samples")
end

function initialize_dataset(
    output_path::String,
    target_samples::Int,
    random_seed::Int,
    num_gens::Int,
)
    isfile(output_path) && error("Output already exists: $(output_path)")

    checkpoint_path = output_path * ".partial.npz"
    rng_path = output_path * ".partial.rng"
    has_checkpoint = isfile(checkpoint_path)
    has_rng = isfile(rng_path)
    has_checkpoint == has_rng ||
        error("Incomplete checkpoint state. Expected both $(checkpoint_path) and $(rng_path)")

    x_demand = zeros(Float32, target_samples, NUM_HOURS, 1)
    y_power = zeros(Float32, target_samples, NUM_HOURS, num_gens)
    y_status = zeros(Float32, target_samples, NUM_HOURS, num_gens)
    daily_cost = zeros(Float64, target_samples)

    if !has_checkpoint
        return (
            rng=MersenneTwister(random_seed),
            x_demand=x_demand,
            y_power=y_power,
            y_status=y_status,
            daily_cost=daily_cost,
            accepted=0,
            attempts=0,
        )
    end

    checkpoint = NPZ.npzread(checkpoint_path)
    accepted = Int(only(checkpoint["accepted_samples"]))
    attempts = Int(only(checkpoint["generation_attempts"]))
    saved_target = Int(only(checkpoint["target_samples"]))
    saved_seed = Int(only(checkpoint["random_seed"]))
    saved_target == target_samples ||
        error("Checkpoint target $(saved_target) does not match requested target $(target_samples)")
    saved_seed == random_seed ||
        error("Checkpoint seed $(saved_seed) does not match requested seed $(random_seed)")

    x_demand[1:accepted, :, :] .= checkpoint["X_demand"]
    y_power[1:accepted, :, :] .= checkpoint["Y_power"]
    y_status[1:accepted, :, :] .= checkpoint["Y_status"]
    daily_cost[1:accepted] .= checkpoint["daily_cost"]
    rng = open(deserialize, rng_path)
    log_message("Resuming checkpoint with $(accepted)/$(target_samples) samples after $(attempts) attempts")

    return (
        rng=rng,
        x_demand=x_demand,
        y_power=y_power,
        y_status=y_status,
        daily_cost=daily_cost,
        accepted=accepted,
        attempts=attempts,
    )
end

function force_initial_minimum_times!(model::Model, u, specs::DataFrame)
    num_gens, num_hours = size(u)
    for g in 1:num_gens
        if specs.InitStatus[g] == 1
            remaining_hours = ceil(Int, max(0.0, specs.MinTU[g] - specs.PastHours[g]))
            for t in 1:min(num_hours, remaining_hours)
                @constraint(model, u[g, t] == 1)
            end
        else
            remaining_hours = ceil(Int, max(0.0, specs.MinTD[g] - specs.PastHours[g]))
            for t in 1:min(num_hours, remaining_hours)
                @constraint(model, u[g, t] == 0)
            end
        end
    end
end

function forbid_unverifiable_terminal_transitions!(model::Model, v, w, specs::DataFrame)
    num_gens, num_hours = size(v)
    for g in 1:num_gens
        min_up = Int(specs.MinTU[g])
        min_down = Int(specs.MinTD[g])
        if min_up > 1
            for t in (num_hours - min_up + 2):num_hours
                @constraint(model, v[g, t] == 0)
            end
        end
        if min_down > 1
            for t in (num_hours - min_down + 2):num_hours
                @constraint(model, w[g, t] == 0)
            end
        end
    end
end

function solve_uc_scenario(
    demand::Vector{Float64},
    specs::DataFrame,
    mip_gap::Float64,
    gurobi_env::Gurobi.Env,
)
    num_gens = nrow(specs)
    num_hours = length(demand)
    num_hours == NUM_HOURS || error("Expected $(NUM_HOURS) demand values")

    model = Model(() -> Gurobi.Optimizer(gurobi_env))
    set_silent(model)
    set_optimizer_attribute(model, "MIPGap", mip_gap)

    @variable(model, p[1:num_gens, 1:num_hours] >= 0.0)
    @variable(model, u[1:num_gens, 1:num_hours], Bin)
    @variable(model, v[1:num_gens, 1:num_hours], Bin)
    @variable(model, w[1:num_gens, 1:num_hours], Bin)

    @objective(
        model,
        Min,
        sum(
            specs.SlopeVarCost[g] * p[g, t] +
            specs.InterVarCost[g] * u[g, t] +
            specs.SUcost1[g] * v[g, t] +
            specs.ShutdownCost[g] * w[g, t]
            for g in 1:num_gens, t in 1:num_hours
        ),
    )

    for t in 1:num_hours
        @constraint(model, sum(p[g, t] for g in 1:num_gens) == demand[t])
        for g in 1:num_gens
            previous_status = t == 1 ? specs.InitStatus[g] : u[g, t - 1]
            previous_power = t == 1 ? specs.IniProd[g] : p[g, t - 1]

            @constraint(model, specs.MinProd[g] * u[g, t] <= p[g, t])
            @constraint(model, p[g, t] <= specs.MaxProd[g] * u[g, t])
            @constraint(model, u[g, t] - previous_status == v[g, t] - w[g, t])
            @constraint(model, v[g, t] + w[g, t] <= 1)

            @constraint(
                model,
                p[g, t] - previous_power <=
                specs.RampUp[g] * previous_status + specs.SUcap[g] * v[g, t],
            )
            @constraint(
                model,
                previous_power - p[g, t] <=
                specs.RampDw[g] * u[g, t] + specs.SDcap[g] * w[g, t],
            )

            min_up = Int(specs.MinTU[g])
            min_down = Int(specs.MinTD[g])
            if min_up > 0
                @constraint(model, sum(v[g, tau] for tau in max(1, t - min_up + 1):t) <= u[g, t])
            end
            if min_down > 0
                @constraint(model, sum(w[g, tau] for tau in max(1, t - min_down + 1):t) <= 1 - u[g, t])
            end
        end
    end

    force_initial_minimum_times!(model, u, specs)
    forbid_unverifiable_terminal_transitions!(model, v, w, specs)
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL || !has_values(model)
        return nothing
    end

    return (
        power=value.(p),
        status=value.(u),
        daily_cost=objective_value(model),
    )
end

function main()
    target_samples = env_int("NUM_SAMPLES", 50_000)
    max_attempts = env_int("MAX_ATTEMPTS", target_samples * 3)
    random_seed = env_int("RANDOM_SEED", 42)
    mip_gap = env_float("MIP_GAP", 1e-3)
    checkpoint_every = env_int("CHECKPOINT_EVERY", 1_000)
    specs_path = env_path("GENERATOR_SPECS", "generator_specs.csv")
    output_path = env_path("OUTPUT_PATH", "uc_new_data_strict.npz")

    specs = load_specs(specs_path)
    base_demand = base_demand_profile()
    num_gens = nrow(specs)
    gurobi_env = Gurobi.Env()
    dataset = initialize_dataset(output_path, target_samples, random_seed, num_gens)
    rng = dataset.rng
    x_demand = dataset.x_demand
    y_power = dataset.y_power
    y_status = dataset.y_status
    daily_cost = dataset.daily_cost
    accepted = dataset.accepted
    attempts = dataset.attempts

    log_message("Generating $(target_samples) strict UC samples")
    log_message("Generator specs: $(specs_path)")
    log_message("Output path: $(output_path)")
    log_message("Random seed: $(random_seed), MIP gap: $(mip_gap)")
    while accepted < target_samples && attempts < max_attempts
        attempts += 1
        demand = sample_demand(rng, base_demand, accepted + 1, target_samples)
        solution = solve_uc_scenario(demand, specs, mip_gap, gurobi_env)
        if solution === nothing
            attempts % 100 == 0 && log_message("Attempt $(attempts): rejected infeasible scenario")
            continue
        end

        accepted += 1
        x_demand[accepted, :, 1] .= Float32.(demand)
        y_power[accepted, :, :] .= Float32.(permutedims(solution.power, (2, 1)))
        y_status[accepted, :, :] .= Float32.(permutedims(solution.status, (2, 1)))
        daily_cost[accepted] = solution.daily_cost

        if accepted % 100 == 0 || accepted == target_samples
            log_message("Accepted $(accepted)/$(target_samples) samples after $(attempts) attempts")
        end
        if accepted % checkpoint_every == 0 && accepted < target_samples
            save_checkpoint!(
                output_path,
                x_demand,
                y_power,
                y_status,
                daily_cost,
                accepted,
                attempts,
                target_samples,
                random_seed,
                mip_gap,
                rng,
            )
        end
    end

    accepted == target_samples ||
        error("Generated only $(accepted) feasible samples after $(attempts) attempts")

    mkpath(dirname(output_path))
    write_dataset(
        output_path,
        x_demand,
        y_power,
        y_status,
        daily_cost,
        accepted,
        attempts,
        target_samples,
        random_seed,
        mip_gap,
    )
    rm(output_path * ".partial.npz"; force=true)
    rm(output_path * ".partial.rng"; force=true)
    log_message("Saved strict UC dataset to $(output_path)")
end

main()
