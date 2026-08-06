from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_CLT_MIN_N = 100  # threshold below which the Central Limit Theorem is unreliable


@dataclass(frozen=True)
class MIWithCI:
    """Mitotic Index with propagated uncertainty (95.45% CI = MI +/- 2 sigma)."""

    mi: float
    var_mi: float
    sigma_mi: float
    ci_lower: float
    ci_upper: float
    n_cel: float
    n_mit: float
    var_cel: float
    var_mit: float


def compute_mi_with_ci(p_hat: np.ndarray, q_hat: np.ndarray) -> MIWithCI:
    """Compute the Mitotic Index with a 95.45% confidence interval.

    Propagates uncertainty from a two-stage detector + classifier pipeline using
    the Delta Method, assuming each detection contributes an independent
    Bernoulli "is-a-real-cell" variable and an independent Bernoulli
    "is-mitotic | is-a-real-cell" variable.

    Parameters
    ----------
    p_hat : np.ndarray of shape (N,)
        Calibrated detection probabilities. p_hat[i] = P(detection i is a real cell).
    q_hat : np.ndarray of shape (N, 2)
        Calibrated classification probabilities. q_hat[i, 0] = P(interphase),
        q_hat[i, 1] = P(mitosis). Rows must sum to 1.

    Returns
    -------
    MIWithCI
        Dataclass with fields ``mi``, ``var_mi``, ``sigma_mi``, ``ci_lower``,
        ``ci_upper``, ``n_cel``, ``n_mit``, ``var_cel``, ``var_mit``. All NaN
        when N_cel is too small to define MI.

    Warns
    -----
    UserWarning
        If N < 100 (Central Limit Theorem assumption likely violated).
    UserWarning
        If N_cel is below floating-point epsilon (returns NaN-filled result).

    Raises
    ------
    ValueError
        If shapes are inconsistent or any probability is outside [0, 1].

    Notes
    -----
    The covariance term in the variance of the ratio uses the identity
    ``cov(N_mit, N_cel) = var_mit`` derived under the independence assumption
    on the per-detection Bernoulli variables.
    """
    # ---- Input validation ----
    p_hat = np.asarray(p_hat, dtype=np.float64)
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if p_hat.ndim != 1:
        raise ValueError(f"p_hat must be 1-D, got shape {p_hat.shape}")
    if q_hat.ndim != 2 or q_hat.shape[1] != 2:
        raise ValueError(f"q_hat must have shape (N, 2), got {q_hat.shape}")
    if p_hat.shape[0] != q_hat.shape[0]:
        raise ValueError(
            f"p_hat and q_hat must have matching N; got {p_hat.shape[0]} vs {q_hat.shape[0]}"
        )
    if np.any((p_hat < 0.0) | (p_hat > 1.0)):
        raise ValueError("p_hat contains values outside [0, 1]")
    if np.any((q_hat < 0.0) | (q_hat > 1.0)):
        raise ValueError("q_hat contains values outside [0, 1]")

    N = p_hat.shape[0]
    if N < _CLT_MIN_N:
        warnings.warn(
            f"N={N} < {_CLT_MIN_N}: CLT assumption underlying the 95.45% CI may not hold.",
            stacklevel=2,
        )

    # ---- Step 1: Expected cell counts from detector ----
    # N_cel = sum_i p_hat[i]
    N_cel = float(np.sum(p_hat))
    # var_cel = sum_i p_hat[i] * (1 - p_hat[i])
    var_cel = float(np.sum(p_hat * (1.0 - p_hat)))

    # ---- Guard: N_cel ~ 0 -> MI is undefined ----
    if N_cel < np.finfo(np.float64).eps:
        warnings.warn(
            "N_cel is effectively zero; Mitotic Index is undefined. Returning NaN result.",
            stacklevel=2,
        )
        nan = float("nan")
        return MIWithCI(
            mi=nan,
            var_mi=nan,
            sigma_mi=nan,
            ci_lower=nan,
            ci_upper=nan,
            n_cel=N_cel,
            n_mit=0.0,
            var_cel=var_cel,
            var_mit=0.0,
        )

    # ---- Step 2: Expected mitotic count from classifier ----
    q_mit = q_hat[:, 1]
    # N_mit = sum_j p_hat[j] * q_hat[j, 1]
    N_mit = float(np.sum(p_hat * q_mit))
    # var_mit = sum_j p_hat[j]**2 * q_hat[j, 1] * (1 - q_hat[j, 1])
    var_mit = float(np.sum((p_hat**2) * q_mit * (1.0 - q_mit)))

    # ---- Step 3: Mitotic Index point estimate ----
    # MI = N_mit / N_cel
    MI = N_mit / N_cel

    # ---- Step 4: Variance of MI via the Delta Method ----
    # Special case: N_mit ~ 0 -> MI = 0 and var_MI = 0 (avoid 0/0 in the formula).
    if N_mit < np.finfo(np.float64).eps:
        var_MI = 0.0
    else:
        # var_MI = MI**2 * (var_mit / N_mit**2 + var_cel / N_cel**2 - 2 * var_mit / (N_mit * N_cel))
        var_MI = (MI**2) * (
            var_mit / (N_mit**2) + var_cel / (N_cel**2) - 2.0 * var_mit / (N_mit * N_cel)
        )
        # Numerical floor: clamp tiny negatives that arise from float roundoff in the bracket.
        if var_MI < 0.0:
            var_MI = 0.0

    # ---- Step 5: 95.45% confidence interval (CLT, k=2) ----
    # sigma_MI = sqrt(var_MI)
    sigma_MI = float(np.sqrt(var_MI))
    # CI_lower = MI - 2 * sigma_MI
    CI_lower = MI - 2.0 * sigma_MI
    # CI_upper = MI + 2 * sigma_MI
    CI_upper = MI + 2.0 * sigma_MI

    return MIWithCI(
        mi=MI,
        var_mi=float(var_MI),
        sigma_mi=sigma_MI,
        ci_lower=CI_lower,
        ci_upper=CI_upper,
        n_cel=N_cel,
        n_mit=N_mit,
        var_cel=var_cel,
        var_mit=float(var_mit),
    )


if __name__ == "__main__":
    # ---- Standalone synthetic validation ----
    # N=200 detections, all certain (p=1.0), each with q_mit=0.1.
    # Expected: N_cel = 200, var_cel = 0, N_mit = 20, var_mit = 200 * 1 * 0.1 * 0.9 = 18.0
    # MI = 0.1; var_MI = 0.1**2 * (18/400 + 0/40000 - 2*18/(20*200)) = 0.01 * (0.045 - 0.009) = 0.00036
    # sigma_MI = sqrt(0.00036) ~= 0.018974
    rng = np.random.default_rng(0)  # noqa: F841
    N = 200
    p = np.ones(N)
    q = np.zeros((N, 2))
    q[:, 0] = 0.9
    q[:, 1] = 0.1
    r = compute_mi_with_ci(p, q)

    assert np.isclose(r.n_cel, 200.0), r
    assert np.isclose(r.var_cel, 0.0), r
    assert np.isclose(r.n_mit, 20.0), r
    assert np.isclose(r.var_mit, 18.0), r
    assert np.isclose(r.mi, 0.1), r
    assert np.isclose(r.var_mi, 0.00036, atol=1e-9), r
    assert np.isclose(r.sigma_mi, np.sqrt(0.00036), atol=1e-9), r
    print("PASS:", r)
