"""Cost-aware allocation model with look-ahead startup repair."""

import numpy as np
import tensorflow as tf

from rnncell_strict_allocation_cost_aware_model import CostAwareLookAheadRepairUCCell
from rnncell_strict_allocation_lookahead_repair_model import FutureDemandWindow
from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model


@tf.keras.utils.register_keras_serializable(package="AIUC")
class LookAheadStartupRepairUCCell(CostAwareLookAheadRepairUCCell):
    """Start eligible offline units early when future ramp capacity is short."""

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
        was_offline = tf.logical_not(was_online)
        initial_hard_status = tf.round(u_final)
        hard_status = initial_hard_status

        startup_allowed = tf.logical_and(
            was_offline,
            tf.logical_and(
                hoff_prev >= self.t_down,
                remaining_hours >= self.t_up,
            ),
        )
        add_allowed = tf.logical_and(
            hard_status < 0.5,
            tf.logical_or(was_online, startup_allowed),
        )
        hard_status = self._vectorized_add_until_upper_bound_covers_demand(
            demand_mw,
            u_ai,
            hard_status,
            add_allowed,
            p_lower,
            p_upper,
        )
        hard_status = self._vectorized_add_until_lookahead_upper_covers_demand(
            demand_mw,
            u_ai,
            hard_status,
            add_allowed,
            cell_inputs=cell_inputs,
            p_prev=p_prev,
            was_online=was_online,
            hoff_prev=hoff_prev,
            remaining_hours=remaining_hours,
        )
        hard_status = self._prepare_status_for_removal(hard_status, was_online)

        shutdown_allowed = tf.logical_and(
            was_online,
            tf.logical_and(
                hon_prev >= self.t_up,
                tf.logical_and(
                    p_prev <= self.sd_cap,
                    remaining_hours >= self.t_down,
                ),
            ),
        )
        remove_allowed = tf.logical_and(
            hard_status > 0.5,
            tf.logical_or(was_offline, shutdown_allowed),
        )
        hard_status = self._vectorized_remove_low_score_prefix(
            demand_mw,
            u_ai,
            hard_status,
            remove_allowed,
            p_lower,
            p_upper,
            cell_inputs=cell_inputs,
            p_prev=p_prev,
            was_online=was_online,
            hoff_prev=hoff_prev,
            remaining_hours=remaining_hours,
        )

        repaired_status = u_final + tf.stop_gradient(hard_status - u_final)
        was_repaired = tf.abs(hard_status - initial_hard_status) > 0.5
        repaired_output = tf.where(was_repaired, hard_status, u_masked)
        output_status = u_masked + tf.stop_gradient(repaired_output - u_masked)
        return repaired_status, output_status

    def _vectorized_add_until_lookahead_upper_covers_demand(
        self,
        demand_mw,
        u_ai,
        status,
        allowed,
        *,
        cell_inputs,
        p_prev,
        was_online,
        hoff_prev,
        remaining_hours,
    ):
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

        candidate = tf.logical_and(allowed, status < 0.5)
        order = tf.argsort(u_ai, axis=-1, direction="DESCENDING")
        ordered_candidate = tf.gather(candidate, order, axis=1, batch_dims=1)
        gain = tf.maximum(kept_upper - restart_upper, 0.0)
        ordered_gain = tf.transpose(
            tf.gather(gain, order, axis=2, batch_dims=1),
            [0, 2, 1],
        )
        gain_before = tf.math.cumsum(
            ordered_gain * tf.cast(ordered_candidate[:, :, tf.newaxis], tf.float32),
            axis=1,
            exclusive=True,
        )
        needs_candidate = tf.reduce_any(
            total_upper[:, tf.newaxis, :] + gain_before
            < demand_window[:, tf.newaxis, :],
            axis=-1,
        )
        ordered_selected = tf.logical_and(ordered_candidate, needs_candidate)
        selected = self._restore_generator_order(
            tf.cast(ordered_selected, tf.float32), order
        )
        return tf.maximum(status, selected)


def build_hybrid_uc_strict_allocation_startup_repair_model(
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
        cell_class=LookAheadStartupRepairUCCell,
        cell_name="lookahead_startup_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_startup_repair",
        cell_kwargs={
            "lookahead_hours": lookahead_hours,
            "lookahead_safety_margin_mw": lookahead_safety_margin_mw,
            "linear_cost_vals": specs["linear_cost"],
        },
        extra_decoder_context_fn=make_extra_decoder_context,
    )
