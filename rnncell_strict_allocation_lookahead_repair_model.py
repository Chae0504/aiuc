"""Strict allocation model with vectorized look-ahead commitment repair."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model
from rnncell_strict_allocation_repair_model import MutMdtAwareCommitmentRepairUCCell


@tf.keras.utils.register_keras_serializable(package="AIUC")
class FutureDemandWindow(Layer):
    """Expose the next K demand values at each decoder step."""

    def __init__(self, lookahead_hours, **kwargs):
        super().__init__(**kwargs)
        self.lookahead_hours = int(lookahead_hours)
        if self.lookahead_hours < 1:
            raise ValueError("lookahead_hours must be positive")

    def call(self, inputs):
        demand_mw = inputs[:, :, 0]
        padded = tf.pad(demand_mw, [[0, 0], [0, self.lookahead_hours]])
        windows = tf.signal.frame(
            padded,
            frame_length=self.lookahead_hours + 1,
            frame_step=1,
            axis=1,
        )
        return windows[:, :, 1:]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.lookahead_hours)

    def get_config(self):
        return {**super().get_config(), "lookahead_hours": self.lookahead_hours}


@tf.keras.utils.register_keras_serializable(package="AIUC")
class LookAheadCommitmentRepairUCCell(MutMdtAwareCommitmentRepairUCCell):
    """Block shutdowns that would create ramp-aware shortage before restart."""

    def __init__(self, lookahead_hours, lookahead_safety_margin_mw=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lookahead_hours = int(lookahead_hours)
        if self.lookahead_hours < 1:
            raise ValueError("lookahead_hours must be positive")
        self.lookahead_safety_margin_mw = float(lookahead_safety_margin_mw)
        if self.lookahead_safety_margin_mw < 0.0:
            raise ValueError("lookahead_safety_margin_mw must be non-negative")
        self.lookahead_safety_margin = tf.constant(
            self.lookahead_safety_margin_mw, dtype=tf.float32
        )

    def _prepare_status_for_removal(self, status, was_online):
        """Reconsider AI-proposed shutdowns under the future-capacity check."""
        return tf.maximum(status, tf.cast(was_online, tf.float32))

    def _vectorized_remove_low_score_prefix(
        self,
        demand_mw,
        u_ai,
        status,
        allowed,
        p_lower,
        p_upper,
        *,
        cell_inputs,
        p_prev,
        was_online,
        hoff_prev,
        remaining_hours,
    ):
        del p_upper
        order = tf.argsort(u_ai, axis=-1, direction="ASCENDING")
        candidate = tf.logical_and(allowed, status > 0.5)
        ordered_candidate = tf.gather(candidate, order, axis=1, batch_dims=1)

        current_lower = tf.reduce_sum(p_lower * status, axis=-1)
        ordered_lower = tf.gather(p_lower, order, axis=1, batch_dims=1)
        candidate_lower = ordered_lower * tf.cast(ordered_candidate, tf.float32)
        removed_lower_before = tf.math.cumsum(
            candidate_lower, axis=-1, exclusive=True
        )

        demand_window = self._demand_window(demand_mw, cell_inputs)
        kept_upper = self._upper_if_kept_online(p_prev, was_online)
        restart_upper = self._upper_after_earliest_restart(
            was_online, hoff_prev, remaining_hours
        )
        online = status > 0.5
        total_upper = tf.reduce_sum(
            tf.where(online[:, tf.newaxis, :], kept_upper, restart_upper),
            axis=-1,
        )
        lost_upper = tf.maximum(kept_upper - restart_upper, 0.0)
        ordered_lost_upper = tf.transpose(
            tf.gather(lost_upper, order, axis=2, batch_dims=1),
            [0, 2, 1],
        )
        removed_upper = tf.math.cumsum(
            ordered_lost_upper
            * tf.cast(ordered_candidate[:, :, tf.newaxis], tf.float32),
            axis=1,
        )
        preserves_lookahead_capacity = tf.reduce_all(
            total_upper[:, tf.newaxis, :] - removed_upper
            >= demand_window[:, tf.newaxis, :],
            axis=-1,
        )
        ordered_selected = tf.logical_and(
            ordered_candidate,
            tf.logical_and(
                removed_lower_before < (current_lower - demand_mw)[:, tf.newaxis],
                preserves_lookahead_capacity,
            ),
        )
        selected = self._restore_generator_order(
            tf.cast(ordered_selected, tf.float32), order
        )
        return status * (1.0 - selected)

    def _upper_if_kept_online(self, p_prev, was_online):
        offsets = self._lookahead_offsets()
        stay_online_upper = tf.minimum(
            self.p_max,
            p_prev[:, tf.newaxis, :] + (offsets + 1.0) * self.ru,
        )
        startup_upper = tf.minimum(
            self.p_max,
            self.su_cap + offsets * self.ru,
        )
        return tf.where(was_online[:, tf.newaxis, :], stay_online_upper, startup_upper)

    def _upper_after_earliest_restart(self, was_online, hoff_prev, remaining_hours):
        offsets = self._lookahead_offsets()
        prior_down_hours = tf.where(was_online, tf.zeros_like(hoff_prev), hoff_prev)
        earliest_restart = tf.maximum(
            1.0,
            self.t_down - prior_down_hours,
        )[:, tf.newaxis, :]
        can_restart = tf.logical_and(
            offsets >= earliest_restart,
            remaining_hours[:, tf.newaxis, :] - earliest_restart
            >= self.t_up,
        )
        ramp_hours = tf.maximum(offsets - earliest_restart, 0.0)
        restart_upper = tf.minimum(
            self.p_max,
            self.su_cap + ramp_hours * self.ru,
        )
        return tf.where(can_restart, restart_upper, 0.0)

    def _demand_window(self, demand_mw, cell_inputs):
        return (
            tf.concat(
                [
                    demand_mw[:, tf.newaxis],
                    cell_inputs[:, 2 : 2 + self.lookahead_hours],
                ],
                axis=-1,
            )
            + self.lookahead_safety_margin
        )

    def _lookahead_offsets(self):
        return tf.cast(
            tf.range(self.lookahead_hours + 1)[tf.newaxis, :, tf.newaxis],
            tf.float32,
        )

    def get_config(self):
        return {
            **super().get_config(),
            "lookahead_hours": self.lookahead_hours,
            "lookahead_safety_margin_mw": self.lookahead_safety_margin_mw,
        }


def build_hybrid_uc_strict_allocation_lookahead_repair_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
    lookahead_hours=None,
    lookahead_safety_margin_mw=0.0,
):
    if lookahead_hours is None:
        lookahead_hours = int(np.max(specs["mdt"]))

    def make_extra_decoder_context(input_dynamic):
        return [
            FutureDemandWindow(
                lookahead_hours,
                name="future_demand_window",
            )(input_dynamic)
        ]

    return build_hybrid_uc_strict_allocation_model(
        specs,
        demand_normalizer_mw=demand_normalizer_mw,
        balance_loss_weight=balance_loss_weight,
        num_hours=num_hours,
        num_static_features=num_static_features,
        cell_class=LookAheadCommitmentRepairUCCell,
        cell_name="lookahead_commitment_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_lookahead_repair",
        cell_kwargs={
            "lookahead_hours": lookahead_hours,
            "lookahead_safety_margin_mw": lookahead_safety_margin_mw,
        },
        extra_decoder_context_fn=make_extra_decoder_context,
    )
