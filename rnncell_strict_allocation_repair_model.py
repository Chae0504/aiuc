"""Strict allocation model with MUT/MDT-aware greedy commitment repair."""

import tensorflow as tf

from rnncell_strict_allocation_model import (
    StrictRampAwareAllocationUCCell,
    build_hybrid_uc_strict_allocation_model,
)


@tf.keras.utils.register_keras_serializable(package="AIUC")
class MutMdtAwareCommitmentRepairUCCell(StrictRampAwareAllocationUCCell):
    """Repair commitment only through transitions allowed by the UC constraints."""

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

    def _prepare_status_for_removal(self, status, was_online):
        """Provide an override point for variants that reconsider AI shutdowns."""
        del was_online
        return status

    def _vectorized_add_until_upper_bound_covers_demand(
        self,
        demand_mw,
        u_ai,
        status,
        allowed,
        p_lower,
        p_upper,
    ):
        del p_lower
        order = tf.argsort(u_ai, axis=-1, direction="DESCENDING")
        total_upper = tf.reduce_sum(p_upper * status, axis=-1)
        ordered_allowed = tf.gather(allowed, order, axis=1, batch_dims=1)
        ordered_upper = tf.gather(p_upper, order, axis=1, batch_dims=1)
        candidate_upper = ordered_upper * tf.cast(ordered_allowed, tf.float32)
        upper_before = tf.math.cumsum(candidate_upper, axis=-1, exclusive=True)
        ordered_selected = tf.logical_and(
            ordered_allowed,
            total_upper[:, tf.newaxis] + upper_before < demand_mw[:, tf.newaxis],
        )
        selected = self._restore_generator_order(
            tf.cast(ordered_selected, tf.float32), order
        )
        return tf.maximum(status, selected)

    def _vectorized_remove_low_score_prefix(
        self,
        demand_mw,
        u_ai,
        status,
        allowed,
        p_lower,
        p_upper,
        **unused_context,
    ):
        del unused_context
        order = tf.argsort(u_ai, axis=-1, direction="ASCENDING")
        total_lower = tf.reduce_sum(p_lower * status, axis=-1)
        total_upper = tf.reduce_sum(p_upper * status, axis=-1)
        candidate = tf.logical_and(allowed, status > 0.5)
        ordered_candidate = tf.gather(candidate, order, axis=1, batch_dims=1)
        ordered_lower = tf.gather(p_lower, order, axis=1, batch_dims=1)
        ordered_upper = tf.gather(p_upper, order, axis=1, batch_dims=1)
        candidate_lower = ordered_lower * tf.cast(ordered_candidate, tf.float32)
        candidate_upper = ordered_upper * tf.cast(ordered_candidate, tf.float32)
        removed_lower_before = tf.math.cumsum(
            candidate_lower, axis=-1, exclusive=True
        )
        removed_upper = tf.math.cumsum(candidate_upper, axis=-1)
        ordered_selected = tf.logical_and(
            ordered_candidate,
            tf.logical_and(
                removed_lower_before
                < (total_lower - demand_mw)[:, tf.newaxis],
                removed_upper <= (total_upper - demand_mw)[:, tf.newaxis],
            ),
        )
        selected = self._restore_generator_order(
            tf.cast(ordered_selected, tf.float32), order
        )
        return status * (1.0 - selected)

    @staticmethod
    def _restore_generator_order(ordered_values, order):
        inverse_order = tf.argsort(order, axis=-1)
        return tf.gather(ordered_values, inverse_order, axis=1, batch_dims=1)


def build_hybrid_uc_strict_allocation_repair_model(
    specs,
    demand_normalizer_mw,
    balance_loss_weight,
    num_hours=24,
    num_static_features=4,
):
    return build_hybrid_uc_strict_allocation_model(
        specs,
        demand_normalizer_mw=demand_normalizer_mw,
        balance_loss_weight=balance_loss_weight,
        num_hours=num_hours,
        num_static_features=num_static_features,
        cell_class=MutMdtAwareCommitmentRepairUCCell,
        cell_name="mut_mdt_aware_commitment_repair_uc_cell",
        model_name="physics_informed_uc_rnn_strict_allocation_repair",
    )
