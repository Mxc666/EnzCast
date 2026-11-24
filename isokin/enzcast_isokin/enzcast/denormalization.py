from __future__ import annotations

import numpy as np


def denormalize_prediction(
    mu_norm,
    logvar_norm,
    task_mean: float,
    task_std: float,
    clamp: tuple[float, float] = (-8.0, 8.0),
):

    mu_arr = np.asarray(mu_norm, dtype=float)
    logvar_arr = np.asarray(logvar_norm, dtype=float)
    clipped_logvar = np.clip(logvar_arr, clamp[0], clamp[1])
    mu_log10 = mu_arr * task_std + task_mean
    std_log10 = np.exp(0.5 * clipped_logvar) * task_std
    return mu_log10, std_log10
