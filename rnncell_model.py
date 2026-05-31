"""TensorFlow model components for the physics-informed UC RNN."""

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


@tf.keras.utils.register_keras_serializable(package="AIUC")
class NormalizedMeanAbsoluteError(tf.keras.losses.Loss):
    """Mean absolute error divided by a domain-specific reference value."""

    def __init__(self, normalizer, name="normalized_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.normalizer = float(normalizer)
        if self.normalizer <= 0:
            raise ValueError("normalizer must be positive")

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1) / self.normalizer

    def get_config(self):
        return {**super().get_config(), "normalizer": self.normalizer}


@tf.keras.utils.register_keras_serializable(package="AIUC")
def ste_binarize(x):
    """Round in the forward pass while keeping a straight-through gradient."""
    return x + tf.stop_gradient(tf.round(x) - x)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class BatchZeros(Layer):
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def call(self, inputs):
        return tf.zeros((tf.shape(inputs)[0], self.width), dtype=inputs.dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.width)

    def get_config(self):
        return {**super().get_config(), "width": self.width}


@tf.keras.utils.register_keras_serializable(package="AIUC")
class InitialDuration(Layer):
    def __init__(self, invert=False, duration=99.0, **kwargs):
        super().__init__(**kwargs)
        self.invert = invert
        self.duration = duration

    def call(self, inputs):
        values = 1.0 - inputs if self.invert else inputs
        return values * self.duration

    def get_config(self):
        return {
            **super().get_config(),
            "invert": self.invert,
            "duration": self.duration,
        }


@tf.keras.utils.register_keras_serializable(package="AIUC")
class PhysicsInformedUCCell(Layer):
    def __init__(
        self,
        hidden_dim,
        num_gens,
        mut_vals,
        mdt_vals,
        p_min_vals,
        p_max_vals,
        ramp_up_vals,
        ramp_down_vals,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_gens = num_gens
        self.mut_vals = list(mut_vals)
        self.mdt_vals = list(mdt_vals)
        self.p_min_vals = list(p_min_vals)
        self.p_max_vals = list(p_max_vals)
        self.ramp_up_vals = list(ramp_up_vals)
        self.ramp_down_vals = list(ramp_down_vals)

        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_dim)
        self.dense_u = Dense(num_gens, activation="sigmoid", name="cell_status_pred")
        self.dense_p = Dense(num_gens, activation="linear", name="cell_power_pred")

        self.t_up = tf.constant(self.mut_vals, dtype=tf.float32)
        self.t_down = tf.constant(self.mdt_vals, dtype=tf.float32)
        self.p_min = tf.constant(self.p_min_vals, dtype=tf.float32)
        self.p_max = tf.constant(self.p_max_vals, dtype=tf.float32)
        self.ru = tf.constant(self.ramp_up_vals, dtype=tf.float32)
        self.rd = tf.constant(self.ramp_down_vals, dtype=tf.float32)

    @property
    def state_size(self):
        return (
            self.hidden_dim,
            self.hidden_dim,
            self.num_gens,
            self.num_gens,
            self.num_gens,
        )

    @property
    def output_size(self):
        return self.num_gens * 2

    def build(self, input_shape):
        batch_size = input_shape[0]
        decoder_width = input_shape[-1] + self.num_gens * 3
        self.lstm_cell.build((batch_size, decoder_width))
        self.dense_u.build((batch_size, self.hidden_dim))
        self.dense_p.build((batch_size, self.hidden_dim))
        super().build(input_shape)

    def call(self, inputs, states):
        h_prev, c_prev, p_prev, hon_prev, hoff_prev = states

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
        p_upper = tf.where(
            hon_prev == 0.0,
            self.p_min + self.ru,
            tf.minimum(self.p_max, p_prev + self.ru),
        )
        p_lower = tf.maximum(self.p_min, p_prev - self.rd)
        p_clip = tf.minimum(tf.maximum(p_raw, p_lower), p_upper)
        p_final = p_clip * u_final

        hon_new = u_final * (hon_prev + 1.0)
        hoff_new = (1.0 - u_final) * (hoff_prev + 1.0)
        new_states = (h_new, c_new, p_final, hon_new, hoff_new)
        output = tf.concat([u_masked, p_final], axis=-1)
        return output, new_states

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "num_gens": self.num_gens,
            "mut_vals": self.mut_vals,
            "mdt_vals": self.mdt_vals,
            "p_min_vals": self.p_min_vals,
            "p_max_vals": self.p_max_vals,
            "ramp_up_vals": self.ramp_up_vals,
            "ramp_down_vals": self.ramp_down_vals,
        }


@tf.keras.utils.register_keras_serializable(package="AIUC")
class OnlyMismatchLoss(Layer):
    def __init__(
        self,
        num_gens,
        demand_normalizer_mw=1.0,
        balance_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_gens = num_gens
        self.demand_normalizer_mw = float(demand_normalizer_mw)
        self.balance_loss_weight = float(balance_loss_weight)
        if self.demand_normalizer_mw <= 0:
            raise ValueError("demand_normalizer_mw must be positive")

    def call(self, inputs):
        demand_mw, rnn_outputs = inputs
        demand_mw = tf.squeeze(demand_mw, axis=-1)
        out_status = rnn_outputs[:, :, : self.num_gens]
        out_power_mw = rnn_outputs[:, :, self.num_gens :]
        pred_total = tf.reduce_sum(out_power_mw, axis=-1)
        mismatch_mae_mw = tf.reduce_mean(tf.abs(pred_total - demand_mw))
        normalized_mismatch = mismatch_mae_mw / self.demand_normalizer_mw
        self.add_loss(self.balance_loss_weight * normalized_mismatch)
        return out_status, out_power_mw

    def set_balance_loss_weight(self, value):
        self.balance_loss_weight = float(value)

    def get_config(self):
        return {
            **super().get_config(),
            "num_gens": self.num_gens,
            "demand_normalizer_mw": self.demand_normalizer_mw,
            "balance_loss_weight": self.balance_loss_weight,
        }


def build_hybrid_uc_model(
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

    custom_cell = PhysicsInformedUCCell(
        hidden_dim=128,
        num_gens=num_gens,
        mut_vals=specs["mut"],
        mdt_vals=specs["mdt"],
        p_min_vals=specs["p_min"],
        p_max_vals=specs["p_max"],
        ramp_up_vals=specs["ramp_up"],
        ramp_down_vals=specs["ramp_down"],
        name="physics_informed_uc_cell",
    )
    rnn_outputs = RNN(
        custom_cell, return_sequences=True, name="physics_rnn_decoder"
    )(
        encoded,
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
        name="physics_informed_uc_rnn",
    )
