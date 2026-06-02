"""Ramp-aware proportional-allocation variant of the physics-informed UC RNN."""

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

from legacy.rnncell_model import (
    BatchZeros,
    InitialDuration,
    OnlyMismatchLoss,
    PhysicsInformedUCCell,
    ste_binarize,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class RampAwareAllocationUCCell(PhysicsInformedUCCell):
    """Allocate residual demand over online generators within ramp-aware bounds."""

    def call(self, inputs, states):
        h_prev, c_prev, p_prev, hon_prev, hoff_prev = states
        demand_mw = inputs[:, 0]

        lstm_input = tf.concat([inputs, p_prev, hon_prev, hoff_prev], axis=-1)
        _, (h_new, c_new) = self.lstm_cell(
            lstm_input, states=[h_prev, c_prev]
        )

        u_ai = self.dense_u(h_new)
        p_ai = self.dense_p(h_new)

        must_on = tf.cast(
            tf.logical_and(hon_prev > 0.0, hon_prev < self.t_up), tf.float32
        )
        must_off = tf.cast(
            tf.logical_and(hoff_prev > 0.0, hoff_prev < self.t_down), tf.float32
        )
        u_masked_hard = u_ai * (1.0 - must_off) + must_on
        u_masked_hard = tf.clip_by_value(u_masked_hard, 0.0, 1.0)
        u_masked = u_ai + tf.stop_gradient(u_masked_hard - u_ai)
        u_final = ste_binarize(u_masked)

        p_raw = self.p_min + tf.nn.sigmoid(p_ai) * (self.p_max - self.p_min)
        startup_upper = tf.minimum(self.p_max, self.p_min + self.ru)
        p_upper = tf.where(
            hon_prev == 0.0,
            startup_upper,
            tf.minimum(self.p_max, p_prev + self.ru),
        )
        p_lower = tf.maximum(self.p_min, p_prev - self.rd)
        p_clip = tf.minimum(tf.maximum(p_raw, p_lower), p_upper) * u_final

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
        allocated_mw = tf.minimum(
            tf.abs(residual_mw[:, tf.newaxis]),
            total_headroom,
        )
        adjustment = (
            tf.math.divide_no_nan(selected_headroom, total_headroom)
            * allocated_mw
            * tf.sign(residual_mw[:, tf.newaxis])
        )
        p_final = p_clip + adjustment
        p_final = tf.minimum(tf.maximum(p_final, p_lower_online), p_upper_online)

        hon_new = u_final * (hon_prev + 1.0)
        hoff_new = (1.0 - u_final) * (hoff_prev + 1.0)
        new_states = (h_new, c_new, p_final, hon_new, hoff_new)
        output = tf.concat([u_masked, p_final], axis=-1)
        return output, new_states


def build_hybrid_uc_allocation_model(
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

    h_0 = BatchZeros(128, name="h_0")(input_dynamic)
    c_0 = BatchZeros(128, name="c_0")(input_dynamic)
    hon_0 = InitialDuration(name="hon_0")(input_init_status)
    hoff_0 = InitialDuration(invert=True, name="hoff_0")(input_init_status)

    static_encoded = Dense(64, activation="relu", name="static_encoder")(
        input_static
    )
    static_repeated = RepeatVector(num_hours, name="static_repeater")(
        static_encoded
    )
    merged_input = Concatenate(axis=-1)([input_dynamic, static_repeated])
    encoded = Bidirectional(
        LSTM(128, return_sequences=True), name="bi_lstm_stack_1"
    )(merged_input)
    encoded = Bidirectional(
        LSTM(128, return_sequences=True), name="bi_lstm_stack_2"
    )(encoded)
    decoder_input = Concatenate(axis=-1, name="decoder_demand_context")(
        [input_dynamic, encoded]
    )

    custom_cell = RampAwareAllocationUCCell(
        hidden_dim=128,
        num_gens=num_gens,
        mut_vals=specs["mut"],
        mdt_vals=specs["mdt"],
        p_min_vals=specs["p_min"],
        p_max_vals=specs["p_max"],
        ramp_up_vals=specs["ramp_up"],
        ramp_down_vals=specs["ramp_down"],
        name="ramp_aware_allocation_uc_cell",
    )
    rnn_outputs = RNN(
        custom_cell, return_sequences=True, name="physics_rnn_decoder"
    )(
        decoder_input,
        initial_state=[h_0, c_0, input_init_power_mw, hon_0, hoff_0],
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
        ],
        outputs=[out_status, out_power],
        name="physics_informed_uc_rnn_allocation",
    )
