"""Cost-aware startup-repair model with one-step ramp-position allocation."""

import numpy as np
import tensorflow as tf

from rnncell_strict_allocation_lookahead_repair_model import FutureDemandWindow
from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model
from rnncell_strict_allocation_startup_repair_model import (
    LookAheadStartupRepairUCCell,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class RampPositionAwareStartupRepairUCCell(LookAheadStartupRepairUCCell):
    """Reposition dispatch so the next-hour demand is reachable under ramp limits."""

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

        next_demand_mw = cell_inputs[:, 2]
        positioned_power = self._ramp_position_dispatch(
            economic_power,
            next_demand_mw,
            p_lower,
            p_upper,
            u_final,
        )
        return p_clip + tf.stop_gradient(positioned_power - p_clip)

    def _ramp_position_dispatch(
        self,
        current_power,
        next_demand_mw,
        p_lower,
        p_upper,
        u_final,
    ):
        p_lower_online = p_lower * u_final
        p_upper_online = p_upper * u_final
        next_upper = tf.minimum(self.p_max, current_power + self.ru) * u_final
        next_shortfall = tf.maximum(
            next_demand_mw - tf.reduce_sum(next_upper, axis=-1),
            0.0,
        )

        receiver_gain = tf.minimum(
            tf.maximum(p_upper_online - current_power, 0.0),
            tf.maximum(self.p_max - (current_power + self.ru), 0.0),
        )
        donor_free = tf.minimum(
            tf.maximum(current_power - p_lower_online, 0.0),
            tf.maximum(current_power + self.ru - self.p_max, 0.0),
        )
        shift_amount = tf.minimum(
            next_shortfall,
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

    @staticmethod
    def _ordered_fill(capacity, amount, order):
        inverse_order = tf.argsort(order)
        ordered_capacity = tf.gather(capacity, order, axis=1)
        used_before = tf.math.cumsum(ordered_capacity, axis=-1, exclusive=True)
        ordered_alloc = tf.minimum(
            tf.maximum(amount[:, tf.newaxis] - used_before, 0.0),
            ordered_capacity,
        )
        return tf.gather(ordered_alloc, inverse_order, axis=1)


def build_hybrid_uc_strict_allocation_ramp_position_model(
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
        cell_class=RampPositionAwareStartupRepairUCCell,
        cell_name="ramp_position_aware_startup_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_ramp_position",
        cell_kwargs={
            "lookahead_hours": lookahead_hours,
            "lookahead_safety_margin_mw": lookahead_safety_margin_mw,
            "linear_cost_vals": specs["linear_cost"],
        },
        extra_decoder_context_fn=make_extra_decoder_context,
    )
