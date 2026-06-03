"""Strict ramp-aware proportional-allocation variant of the UC RNN."""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Bidirectional,
    Concatenate,
    Dense,
    Input,
    Layer,
    LSTM,
    RNN,
    RepeatVector,
)
from tensorflow.keras.models import Model

from legacy.rnncell_model import (
    BatchZeros,
    OnlyMismatchLoss,
    PhysicsInformedUCCell,
    ste_binarize,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class RemainingHours(Layer):
    """Create a descending horizon counter for hard terminal-transition rules."""

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_hours = tf.shape(inputs)[1]
        values = tf.cast(tf.range(num_hours, 0, -1), inputs.dtype)
        return tf.tile(values[tf.newaxis, :, tf.newaxis], [batch_size, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class StrictRampAwareAllocationUCCell(PhysicsInformedUCCell):
    """Enforce hourly UC transition limits and allocate residual demand."""

    def __init__(self, startup_cap_vals, shutdown_cap_vals, **kwargs):
        super().__init__(**kwargs)
        self.startup_cap_vals = list(startup_cap_vals)
        self.shutdown_cap_vals = list(shutdown_cap_vals)
        if len(self.startup_cap_vals) != self.num_gens:
            raise ValueError("startup_cap_vals must contain one value per generator")
        if len(self.shutdown_cap_vals) != self.num_gens:
            raise ValueError("shutdown_cap_vals must contain one value per generator")
        self.su_cap = tf.constant(self.startup_cap_vals, dtype=tf.float32)
        self.sd_cap = tf.constant(self.shutdown_cap_vals, dtype=tf.float32)

    @property
    def state_size(self):
        return (
            self.hidden_dim,
            self.hidden_dim,
            self.num_gens,
            self.num_gens,
            self.num_gens,
            self.num_gens,
        )

    def build(self, input_shape):
        batch_size = input_shape[0]
        decoder_width = input_shape[-1] + self.num_gens * 4
        self.lstm_cell.build((batch_size, decoder_width))
        self.dense_u.build((batch_size, self.hidden_dim))
        self.dense_p.build((batch_size, self.hidden_dim))
        Layer.build(self, input_shape)

    def call(self, inputs, states):
        h_prev, c_prev, p_prev, u_prev, hon_prev, hoff_prev = states
        demand_mw = inputs[:, 0]
        remaining_hours = inputs[:, 1:2]

        lstm_input = tf.concat(
            [inputs, p_prev, u_prev, hon_prev, hoff_prev], axis=-1
        )
        _, (h_new, c_new) = self.lstm_cell(lstm_input, states=[h_prev, c_prev])

        u_ai = self.dense_u(h_new)
        p_ai = self.dense_p(h_new)
        was_online = u_prev > 0.5
        was_offline = tf.logical_not(was_online)

        minimum_up_must_on = tf.logical_and(hon_prev < self.t_up, was_online)
        shutdown_cap_must_on = tf.logical_and(p_prev > self.sd_cap, was_online)
        terminal_must_on = tf.logical_and(remaining_hours < self.t_down, was_online)
        must_on = tf.cast(
            tf.logical_or(
                tf.logical_or(minimum_up_must_on, shutdown_cap_must_on),
                terminal_must_on,
            ),
            tf.float32,
        )

        minimum_down_must_off = tf.logical_and(hoff_prev < self.t_down, was_offline)
        terminal_must_off = tf.logical_and(remaining_hours < self.t_up, was_offline)
        must_off = tf.cast(
            tf.logical_or(minimum_down_must_off, terminal_must_off), tf.float32
        )

        u_masked_hard = u_ai * (1.0 - must_off) + must_on
        u_masked_hard = tf.clip_by_value(u_masked_hard, 0.0, 1.0)
        u_masked = u_ai + tf.stop_gradient(u_masked_hard - u_ai)
        u_final = ste_binarize(u_masked)

        p_raw = self.p_min + tf.nn.sigmoid(p_ai) * (self.p_max - self.p_min)
        p_upper = tf.where(
            was_online,
            tf.minimum(self.p_max, p_prev + self.ru),
            self.su_cap,
        )
        p_lower = tf.where(
            was_online,
            tf.maximum(self.p_min, p_prev - self.rd),
            self.p_min,
        )
        u_final, u_output = self._repair_commitment(
            cell_inputs=inputs,
            demand_mw=demand_mw,
            u_ai=u_ai,
            u_masked=u_masked,
            u_final=u_final,
            p_prev=p_prev,
            was_online=was_online,
            hon_prev=hon_prev,
            hoff_prev=hoff_prev,
            remaining_hours=remaining_hours,
            p_lower=p_lower,
            p_upper=p_upper,
        )
        p_clip = tf.minimum(tf.maximum(p_raw, p_lower), p_upper) * u_final

        p_final = self._finalize_power(
            demand_mw, p_clip, p_lower, p_upper, u_final
        )

        hon_new = u_final * (hon_prev + 1.0)
        hoff_new = (1.0 - u_final) * (hoff_prev + 1.0)
        new_states = (h_new, c_new, p_final, u_final, hon_new, hoff_new)
        output = tf.concat([u_output, p_final], axis=-1)
        return output, new_states

    def _repair_commitment(
        self,
        cell_inputs,
        demand_mw,
        u_ai,
        u_masked,
        u_final,
        p_prev,
        was_online,
        hon_prev,
        hoff_prev,
        remaining_hours,
        p_lower,
        p_upper,
    ):
        """Provide an override point for constraint-preserving status repair."""
        del (
            cell_inputs,
            demand_mw,
            u_ai,
            p_prev,
            was_online,
            hon_prev,
            hoff_prev,
            remaining_hours,
            p_lower,
            p_upper,
        )
        return u_final, u_masked

    def _finalize_power(self, demand_mw, p_clip, p_lower, p_upper, u_final):
        """Allocate residual demand over the physically available headroom."""
        p_upper_online = p_upper * u_final
        p_lower_online = p_lower * u_final
        residual_mw = demand_mw - tf.reduce_sum(p_clip, axis=-1)
        up_headroom = tf.maximum(p_upper_online - p_clip, 0.0)
        down_headroom = tf.maximum(p_clip - p_lower_online, 0.0)
        selected_headroom = tf.where(
            residual_mw[:, tf.newaxis] >= 0.0,
            up_headroom,
            down_headroom,
        )
        total_headroom = tf.reduce_sum(selected_headroom, axis=-1, keepdims=True)
        allocated_mw = tf.minimum(tf.abs(residual_mw[:, tf.newaxis]), total_headroom)
        adjustment = (
            tf.math.divide_no_nan(selected_headroom, total_headroom)
            * allocated_mw
            * tf.sign(residual_mw[:, tf.newaxis])
        )
        p_final = p_clip + adjustment
        return tf.minimum(tf.maximum(p_final, p_lower_online), p_upper_online)

    def get_config(self):
        return {
            **super().get_config(),
            "startup_cap_vals": self.startup_cap_vals,
            "shutdown_cap_vals": self.shutdown_cap_vals,
        }


def build_hybrid_uc_strict_allocation_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
    cell_class=StrictRampAwareAllocationUCCell,
    cell_name="strict_ramp_aware_allocation_uc_cell",
    model_name="physics_informed_uc_rnn_strict_allocation",
    cell_kwargs=None,
    extra_decoder_context_fn=None,
):
    if cell_kwargs is None:
        cell_kwargs = {}
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
    extra_decoder_context = []
    if extra_decoder_context_fn is not None:
        extra_decoder_context = list(extra_decoder_context_fn(input_dynamic))
    decoder_input = Concatenate(axis=-1, name="decoder_demand_context")(
        [input_dynamic, remaining_hours, *extra_decoder_context, encoded]
    )

    custom_cell = cell_class(
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
        name=cell_name,
        **cell_kwargs,
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
        name=model_name,
    )
