from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

LN10 = np.log(10.0)
TASK_ORDER = ("km", "kcat", "kcat_km", "ki")


def compute_complex_rate_limit(log10_expression, complex_index, isoform_mask=None):
    expression = np.asarray(log10_expression, dtype=float)
    if expression.ndim != 3:
        raise ValueError("log10_expression must have shape (contexts, reactions, isoform_slots)")
    n_contexts, n_reactions, n_isoforms = expression.shape
    if len(complex_index) != n_reactions:
        raise ValueError("complex_index must provide one complex list per reaction")

    base_mask = np.ones((n_reactions, n_isoforms), dtype=bool)
    if isoform_mask is not None:
        base_mask = np.asarray(isoform_mask, dtype=bool)
        if base_mask.shape != (n_reactions, n_isoforms):
            raise ValueError("isoform_mask must have shape (reactions, isoform_slots)")

    max_complexes = max((len(rc) for rc in complex_index), default=0)
    limited = np.full((n_contexts, n_reactions, max_complexes), -np.inf, dtype=float)
    complex_mask = np.zeros((n_reactions, max_complexes), dtype=bool)

    for r_idx, reaction_complexes in enumerate(complex_index):
        for c_idx, subunit_indices in enumerate(reaction_complexes):
            indices = tuple(int(i) for i in subunit_indices)
            if not indices:
                continue
            invalid = [i for i in indices if i < 0 or i >= n_isoforms]
            if invalid:
                raise ValueError(f"complex_index has invalid isoform slot(s) for reaction {r_idx}: {invalid}")
            if not all(base_mask[r_idx, i] for i in indices):
                continue
            limited[:, r_idx, c_idx] = np.min(expression[:, r_idx, indices], axis=1)
            complex_mask[r_idx, c_idx] = True
    return limited, complex_mask


def logsumexp_base10(values, axis=None, mask=None):
    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = np.where(mask, arr, -np.inf)
    if axis is None:
        finite = np.isfinite(arr)
        if not finite.any():
            return np.array(-np.inf)
        max_val = np.max(arr[finite])
        return max_val + np.log10(np.sum(np.power(10.0, arr[finite] - max_val)))
    max_val = np.max(arr, axis=axis, keepdims=True)
    all_missing = ~np.isfinite(max_val)
    safe_max = np.where(all_missing, 0.0, max_val)
    stable = np.where(np.isfinite(arr), np.power(10.0, arr - safe_max), 0.0)
    summed = np.sum(stable, axis=axis, keepdims=True)
    log_summed = np.full_like(summed, -np.inf)
    np.log10(summed, out=log_summed, where=summed > 0.0)
    out = np.where(summed > 0.0, safe_max + log_summed, -np.inf)
    return np.squeeze(out, axis=axis) if axis is not None else np.squeeze(out)


def compute_expression_weights(log10_expression, isoform_mask, complex_index=None):
    if complex_index is not None:
        log10_expression, isoform_mask = compute_complex_rate_limit(log10_expression, complex_index, isoform_mask)
    expression = np.power(10.0, np.asarray(log10_expression, dtype=float))
    mask = np.asarray(isoform_mask, dtype=bool)
    expression = np.where(mask[None, :, :], expression, 0.0)
    denom = expression.sum(axis=2, keepdims=True)
    return np.divide(expression, denom, out=np.zeros_like(expression), where=denom > 0)


def compute_effective_kinetics(theta_iso, log10_expression, isoform_mask, complex_index=None):
    theta = np.asarray(theta_iso, dtype=float)
    weights = compute_expression_weights(log10_expression, isoform_mask, complex_index=complex_index)
    if theta.shape[:2] != weights.shape[1:3]:
        raise ValueError("theta_iso reaction/isoform dimensions must match expression aggregation dimensions")
    log_weights = np.full_like(weights, -np.inf)
    np.log10(weights, out=log_weights, where=weights > 0.0)
    terms = log_weights[:, :, :, None] + theta[None, :, :, :]
    mask = (weights > 0.0)[:, :, :, None] & np.isfinite(log_weights[:, :, :, None])
    return logsumexp_base10(terms, axis=2, mask=mask)


def _logsumexp_base10_lastdim(log10_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ln_vals = np.where(mask, log10_vals * LN10, -np.inf)
    return logsumexp(ln_vals, axis=-1) / LN10


def compute_effective_kinetics_draws(
    posterior_samples: dict[str, np.ndarray],
    log10_E_iso: np.ndarray,
    isoform_mask: np.ndarray,
    tasks: tuple[str, ...] = TASK_ORDER,
) -> dict[str, np.ndarray]:
    mask = np.asarray(isoform_mask, dtype=bool)
    e_linear = np.where(mask[np.newaxis, :, :], np.power(10.0, log10_E_iso), 0.0)
    e_sum = np.maximum(e_linear.sum(axis=-1, keepdims=True), 1e-30)
    w_iso = e_linear / e_sum
    G, R, _ = w_iso.shape

    theta_keys = [f"theta_iso_{t}" for t in tasks]
    n_samples = int(posterior_samples[theta_keys[0]].shape[0])
    theta_eff_all = np.empty((n_samples, G, R, len(tasks)), dtype=np.float64)

    log10_w = np.where(mask[np.newaxis, :, :], np.log10(np.maximum(w_iso, 1e-30)), -np.inf)
    for q_idx, key in enumerate(theta_keys):
        theta_q = posterior_samples[key]
        for s_idx in range(n_samples):
            combined = log10_w + theta_q[s_idx][np.newaxis, :, :]
            theta_eff_all[s_idx, :, :, q_idx] = _logsumexp_base10_lastdim(combined, mask[np.newaxis, :, :])

    return {
        "theta_eff_draws": theta_eff_all,
        "theta_eff_mean": theta_eff_all.mean(axis=0),
        "theta_eff_std": theta_eff_all.std(axis=0),
        "theta_eff_q05": np.percentile(theta_eff_all, 5, axis=0),
        "theta_eff_q95": np.percentile(theta_eff_all, 95, axis=0),
        "w_iso": w_iso,
    }


def compute_variance_decomposition(
    posterior_samples: dict[str, np.ndarray],
    log10_E_iso: np.ndarray,
    isoform_mask: np.ndarray,
    reaction_ids: list[str],
    tasks: tuple[str, ...] = TASK_ORDER,
) -> pd.DataFrame:
    mask = np.asarray(isoform_mask, dtype=bool)
    e_linear = np.where(mask[np.newaxis, :, :], np.power(10.0, log10_E_iso), 0.0)
    w_iso = e_linear / np.maximum(e_linear.sum(axis=-1, keepdims=True), 1e-30)
    w_mean = w_iso.mean(axis=0, keepdims=True)
    _, R, _ = w_iso.shape

    rows = []
    for task in tasks:
        theta_q = posterior_samples[f"theta_iso_{task}"]
        n_samples = int(theta_q.shape[0])
        theta_mean = theta_q.mean(axis=0)

        log10_w = np.where(mask[np.newaxis, :, :], np.log10(np.maximum(w_iso, 1e-30)), -np.inf)
        eff_iso = _logsumexp_base10_lastdim(log10_w + theta_mean[np.newaxis, :, :], mask[np.newaxis, :, :])
        var_isoform = eff_iso.var(axis=0)

        log10_w_mean = np.where(mask, np.log10(np.maximum(w_mean[0], 1e-30)), -np.inf)
        eff_kin = np.empty((n_samples, R), dtype=np.float64)
        for s_idx in range(n_samples):
            eff_kin[s_idx] = _logsumexp_base10_lastdim(log10_w_mean + theta_q[s_idx], mask)
        var_kinetic = eff_kin.var(axis=0)

        var_total = var_isoform + var_kinetic
        var_safe = np.maximum(var_total, 1e-30)
        for r_idx, rxn_id in enumerate(reaction_ids):
            rows.append(
                {
                    "reaction_id": rxn_id,
                    "parameter": task,
                    "var_total": float(var_total[r_idx]),
                    "var_isoform_ratio": float(var_isoform[r_idx]),
                    "var_kinetic": float(var_kinetic[r_idx]),
                    "frac_isoform": float(var_isoform[r_idx] / var_safe[r_idx]),
                    "frac_kinetic": float(var_kinetic[r_idx] / var_safe[r_idx]),
                }
            )
    return pd.DataFrame(rows)
