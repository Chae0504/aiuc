"""Cost-aware startup-repair model with multi-step ramp-position allocation."""

import numpy as np
import tensorflow as tf

from rnncell_strict_allocation_lookahead_repair_model import FutureDemandWindow
from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model
from rnncell_strict_allocation_ramp_position_model import (
    RampPositionAwareStartupRepairUCCell,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class MultiStepRampPositionAwareStartupRepairUCCell(
    RampPositionAwareStartupRepairUCCell
):
    """Greedily reposition current dispatch for several future ramp checks."""

    def _finalize_power(
        self,
        demand_mw,
        p_clip,
        p_lower,
        p_upper,
        u_final,
        cell_inputs=None,
    ):
        economic_power = self._merit_order_dispatch(
            demand_mw, p_lower, p_upper, u_final
        )
        if cell_inputs is None:
            return p_clip + tf.stop_gradient(economic_power - p_clip)

        future_demand_mw = cell_inputs[:, 2 : 2 + self.lookahead_hours]
        positioned_power = self._multi_step_ramp_position_dispatch(
            economic_power,
            future_demand_mw,
            p_lower,
            p_upper,
            u_final,
        )
        return p_clip + tf.stop_gradient(positioned_power - p_clip)

    def _multi_step_ramp_position_dispatch(
        self,
        current_power,
        future_demand_mw,
        p_lower,
        p_upper,
        u_final,
    ):
        positioned_power = current_power
        for offset in range(1, self.lookahead_hours + 1):
            positioned_power = self._ramp_position_for_offset(
                positioned_power,
                future_demand_mw[:, offset - 1],
                float(offset),
                p_lower,
                p_upper,
                u_final,
            )
        return positioned_power

    def _ramp_position_for_offset(
        self,
        current_power,
        future_demand_mw,
        offset,
        p_lower,
        p_upper,
        u_final,
    ):
        p_lower_online = p_lower * u_final
        p_upper_online = p_upper * u_final
        offset_upper = tf.minimum(
            self.p_max, current_power + offset * self.ru
        ) * u_final
        shortfall = tf.maximum(
            future_demand_mw - tf.reduce_sum(offset_upper, axis=-1),
            0.0,
        )

        receiver_gain = tf.minimum(
            tf.maximum(p_upper_online - current_power, 0.0),
            tf.maximum(self.p_max - (current_power + offset * self.ru), 0.0),
        )
        donor_free = tf.minimum(
            tf.maximum(current_power - p_lower_online, 0.0),
            self._free_donor_capacity(current_power, offset),
        )
        shift_amount = tf.minimum(
            shortfall,
            tf.minimum(
                tf.reduce_sum(receiver_gain, axis=-1),
                tf.reduce_sum(donor_free, axis=-1),
            ),
        )

        receiver_order = tf.argsort(self.linear_cost, direction="ASCENDING")
        donor_order = tf.argsort(self.linear_cost, direction="DESCENDING")
        receiver_alloc = self._ordered_fill(receiver_gain, shift_amount, receiver_order)
        donor_reduction = self._ordered_fill(donor_free, shift_amount, donor_order)
        positioned_power = current_power + receiver_alloc - donor_reduction
        return tf.minimum(tf.maximum(positioned_power, p_lower_online), p_upper_online)

    def _free_donor_capacity(self, current_power, offset):
        free_capacity = tf.fill(tf.shape(current_power), tf.float32.max)
        for prior_offset in range(1, int(offset) + 1):
            free_capacity = tf.minimum(
                free_capacity,
                tf.maximum(
                    current_power + float(prior_offset) * self.ru - self.p_max,
                    0.0,
                ),
            )
        return free_capacity


def build_hybrid_uc_strict_allocation_multistep_ramp_position_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
    lookahead_hours=None,
    lookahead_safety_margin_mw=0.0,
    status_loss_mode="bce",
    status_false_on_alpha=0.5,
    status_transition_loss_weight=0.5,
    status_online_hours_loss_weight=0.1,
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
        cell_class=MultiStepRampPositionAwareStartupRepairUCCell,
        cell_name="multistep_ramp_position_aware_startup_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_multistep_ramp_position",
        cell_kwargs={
            "lookahead_hours": lookahead_hours,
            "lookahead_safety_margin_mw": lookahead_safety_margin_mw,
            "linear_cost_vals": specs["linear_cost"],
        },
        extra_decoder_context_fn=make_extra_decoder_context,
        status_loss_mode=status_loss_mode,
        status_false_on_alpha=status_false_on_alpha,
        status_transition_loss_weight=status_transition_loss_weight,
        status_online_hours_loss_weight=status_online_hours_loss_weight,
    )
