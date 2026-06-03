"""Strict look-ahead repair model with cost-aware power allocation."""

import numpy as np
import tensorflow as tf

from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model
from rnncell_strict_allocation_lookahead_repair_model import (
    FutureDemandWindow,
    LookAheadCommitmentRepairUCCell,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class CostAwareLookAheadRepairUCCell(LookAheadCommitmentRepairUCCell):
    """Use look-ahead commitment repair, then dispatch online units by cost."""

    def __init__(self, linear_cost_vals, **kwargs):
        super().__init__(**kwargs)
        self.linear_cost_vals = list(linear_cost_vals)
        if len(self.linear_cost_vals) != self.num_gens:
            raise ValueError("linear_cost_vals must contain one value per generator")
        self.linear_cost = tf.constant(self.linear_cost_vals, dtype=tf.float32)

    def _finalize_power(self, demand_mw, p_clip, p_lower, p_upper, u_final):
        economic_power = self._merit_order_dispatch(demand_mw, p_lower, p_upper, u_final)
        return p_clip + tf.stop_gradient(economic_power - p_clip)

    def _merit_order_dispatch(self, demand_mw, p_lower, p_upper, u_final):
        p_lower_online = p_lower * u_final
        p_upper_online = p_upper * u_final
        base_power = p_lower_online
        headroom = tf.maximum(p_upper_online - p_lower_online, 0.0)

        residual = tf.maximum(
            demand_mw - tf.reduce_sum(base_power, axis=-1),
            0.0,
        )
        order = tf.argsort(self.linear_cost, direction="ASCENDING")
        inverse_order = tf.argsort(order)

        ordered_headroom = tf.gather(headroom, order, axis=1)
        used_before = tf.math.cumsum(ordered_headroom, axis=-1, exclusive=True)
        ordered_alloc = tf.minimum(
            tf.maximum(residual[:, tf.newaxis] - used_before, 0.0),
            ordered_headroom,
        )
        allocation = tf.gather(ordered_alloc, inverse_order, axis=1)
        return tf.minimum(tf.maximum(base_power + allocation, p_lower_online), p_upper_online)

    def get_config(self):
        return {**super().get_config(), "linear_cost_vals": self.linear_cost_vals}


def build_hybrid_uc_strict_allocation_cost_aware_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
    lookahead_hours=None,
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
        cell_class=CostAwareLookAheadRepairUCCell,
        cell_name="cost_aware_lookahead_commitment_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_cost_aware",
        cell_kwargs={
            "lookahead_hours": lookahead_hours,
            "linear_cost_vals": specs["linear_cost"],
        },
        extra_decoder_context_fn=make_extra_decoder_context,
    )
