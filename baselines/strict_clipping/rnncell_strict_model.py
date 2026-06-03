"""Strict-clipping baseline variant of the physics-informed UC RNN."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Bidirectional,
    Concatenate,
    Dense,
    Input,
    LSTM,
    RNN,
    RepeatVector,
)
from tensorflow.keras.models import Model

from legacy.rnncell_model import BatchZeros, OnlyMismatchLoss
from rnncell_strict_allocation_model import (
    RemainingHours,
    StrictRampAwareAllocationUCCell,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class StrictClippingUCCell(StrictRampAwareAllocationUCCell):
    """Apply strict hourly UC clipping without residual-demand allocation."""

    def _finalize_power(self, demand_mw, p_clip, p_lower, p_upper, u_final):
        del demand_mw, p_lower, p_upper, u_final
        return p_clip


def build_hybrid_uc_strict_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
):
    num_gens = len(specs["p_max"])
    input_dynamic = Input(shape=(num_hours, 1), name="demand_input")
    input_static = Input(
        shape=(num_gens * num_static_features,), name="static_initial_input"
    )
    input_init_status = Input(shape=(num_gens,), name="input_init_status")
    input_init_power_mw = Input(shape=(num_gens,), name="input_init_power_mw")
    input_init_up_hours = Input(shape=(num_gens,), name="input_init_up_hours")
    input_init_down_hours = Input(shape=(num_gens,), name="input_init_down_hours")

    h_0 = BatchZeros(128, name="h_0")(input_dynamic)
    c_0 = BatchZeros(128, name="c_0")(input_dynamic)
    remaining_hours = RemainingHours(name="remaining_hours")(input_dynamic)

    static_encoded = Dense(64, activation="relu", name="static_encoder")(input_static)
    static_repeated = RepeatVector(num_hours, name="static_repeater")(static_encoded)
    merged_input = Concatenate(axis=-1)([input_dynamic, static_repeated])
    encoded = Bidirectional(
        LSTM(128, return_sequences=True), name="bi_lstm_stack_1"
    )(merged_input)
    encoded = Bidirectional(
        LSTM(128, return_sequences=True), name="bi_lstm_stack_2"
    )(encoded)
    decoder_input = Concatenate(axis=-1, name="decoder_demand_context")(
        [input_dynamic, remaining_hours, encoded]
    )

    custom_cell = StrictClippingUCCell(
        hidden_dim=128,
        num_gens=num_gens,
        mut_vals=specs["mut"],
        mdt_vals=specs["mdt"],
        p_min_vals=specs["p_min"],
        p_max_vals=specs["p_max"],
        ramp_up_vals=specs["ramp_up"],
        ramp_down_vals=specs["ramp_down"],
        startup_cap_vals=specs["startup_cap"],
        shutdown_cap_vals=specs["shutdown_cap"],
        name="strict_clipping_uc_cell",
    )
    rnn_outputs = RNN(
        custom_cell, return_sequences=True, name="physics_rnn_decoder"
    )(
        decoder_input,
        initial_state=[
            h_0,
            c_0,
            input_init_power_mw,
            input_init_status,
            input_init_up_hours,
            input_init_down_hours,
        ],
    )
    status_raw, power_raw = OnlyMismatchLoss(
        num_gens,
        demand_normalizer_mw=demand_normalizer_mw,
        balance_loss_weight=balance_loss_weight,
        name="mismatch_loss_layer",
    )([input_dynamic, rnn_outputs])
    out_status = Activation("linear", name="out_status")(status_raw)
    out_power = Activation("linear", name="out_power")(power_raw)

    return Model(
        inputs=[
            input_dynamic,
            input_static,
            input_init_status,
            input_init_power_mw,
            input_init_up_hours,
            input_init_down_hours,
        ],
        outputs=[out_status, out_power],
        name="physics_informed_uc_rnn_strict",
    )
