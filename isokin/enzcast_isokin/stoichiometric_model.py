from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam
import torch
from torch import Tensor

DETECT_THRESHOLD = 0.05
DETECT_STEEPNESS = 10.0
EPSILON_FLUX = 1e-10
KI_TIGHT_SIGMA = 0.20
SIGMA_S_GLOBAL_PRIOR = 0.20
SIGMA_S_CONTEXT_PRIOR = 0.15
SIGMA_BASE_PRIOR = 0.50
FLUX_LOG10_CLIP_MIN = -6.0
FLUX_LOG10_CLIP_MAX = 6.0
ALPHA_D = 0.30
CONSISTENCY_WEIGHT = 0.10
SCALE_ANCHOR_SIGMA = 0.02


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pyro.set_rng_seed(seed)


def resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def logsumexp_base10(log10_vals: Tensor, mask: Tensor) -> Tensor:
    ln10 = torch.log(torch.tensor(10.0, device=log10_vals.device, dtype=log10_vals.dtype))
    ln_vals = log10_vals * ln10
    ln_vals = torch.where(
        mask,
        ln_vals,
        torch.tensor(-float("inf"), device=ln_vals.device, dtype=ln_vals.dtype),
    )
    return torch.logsumexp(ln_vals, dim=-1) / ln10


def reliability_weights(
    detection_rate: Tensor,
    threshold: float = DETECT_THRESHOLD,
    steepness: float = DETECT_STEEPNESS,
) -> Tensor:
    return torch.sigmoid(steepness * (detection_rate - threshold))


def isokin_stoichiometric_model(
    log10_E_iso: Tensor,
    S: Tensor,
    met_degrees: Tensor,
    prior_mu_iso: Tensor,
    prior_sigma_iso: Tensor,
    isoform_mask: Tensor,
    detection_rate_iso: Tensor,
    use_flux_balance: bool = True,
    detect_threshold: float = DETECT_THRESHOLD,
    detect_steepness: float = DETECT_STEEPNESS,
    sigma_s_global_prior: float = SIGMA_S_GLOBAL_PRIOR,
    sigma_s_context_prior: float = SIGMA_S_CONTEXT_PRIOR,
    sigma_base_prior: float = SIGMA_BASE_PRIOR,
    consistency_weight: float = CONSISTENCY_WEIGHT,
    ki_tight_sigma: float = KI_TIGHT_SIGMA,
    alpha_d: float = ALPHA_D,
    scale_anchor_sigma: float = SCALE_ANCHOR_SIGMA,
) -> dict[str, Tensor]:
    G, R, Imax = log10_E_iso.shape
    device = log10_E_iso.device
    idx_km, idx_kcat, idx_kcat_km, idx_ki = 0, 1, 2, 3
    mask_float = isoform_mask.float()

    with torch.no_grad():
        ref_k_eff = 0.5 * prior_mu_iso[:, :, idx_kcat] + 0.5 * prior_mu_iso[:, :, idx_kcat_km]
        ref_log10_v_iso = (log10_E_iso + ref_k_eff.unsqueeze(0)).clamp(FLUX_LOG10_CLIP_MIN, FLUX_LOG10_CLIP_MAX)
        ref_log10_v = logsumexp_base10(
            ref_log10_v_iso,
            isoform_mask.unsqueeze(0).expand_as(ref_log10_v_iso),
        )
        ref_v = torch.pow(10.0, ref_log10_v)
        ref_throughput = torch.matmul(ref_v, S.abs().mT).clamp(min=EPSILON_FLUX)

    sigma_floor = 1e-4
    safe_sigma = prior_sigma_iso.clamp(min=sigma_floor)

    theta_iso_km = pyro.sample(
        "theta_iso_km",
        dist.Normal(prior_mu_iso[:, :, idx_km], safe_sigma[:, :, idx_km]).to_event(2),
    )
    theta_iso_kcat = pyro.sample(
        "theta_iso_kcat",
        dist.Normal(prior_mu_iso[:, :, idx_kcat], safe_sigma[:, :, idx_kcat]).to_event(2),
    )
    theta_iso_kcat_km = pyro.sample(
        "theta_iso_kcat_km",
        dist.Normal(prior_mu_iso[:, :, idx_kcat_km], safe_sigma[:, :, idx_kcat_km]).to_event(2),
    )
    theta_iso_ki = pyro.sample(
        "theta_iso_ki",
        dist.Normal(
            prior_mu_iso[:, :, idx_ki],
            torch.where(
                isoform_mask,
                torch.full_like(safe_sigma[:, :, idx_ki], ki_tight_sigma),
                torch.full_like(safe_sigma[:, :, idx_ki], sigma_floor),
            ),
        ).to_event(2),
    )

    omega = pyro.sample(
        "omega",
        dist.Beta(torch.full((R,), 2.0, device=device), torch.full((R,), 2.0, device=device)).to_event(1),
    )
    s_base = pyro.sample(
        "s_base",
        dist.Normal(torch.zeros(R, device=device), torch.full((R,), sigma_s_global_prior, device=device)).to_event(1),
    )
    sigma_base = pyro.sample("sigma_base", dist.HalfNormal(torch.tensor(sigma_base_prior, device=device)))
    eps_context = pyro.sample(
        "eps_context",
        dist.Normal(torch.zeros(G, R, device=device), torch.full((G, R), sigma_s_context_prior, device=device)).to_event(2),
    )

    kinetic_iso = omega.unsqueeze(-1) * theta_iso_kcat + (1.0 - omega.unsqueeze(-1)) * theta_iso_kcat_km
    log10_v_iso = (
        log10_E_iso
        + kinetic_iso.unsqueeze(0)
        + s_base.unsqueeze(0).unsqueeze(-1)
        + eps_context.unsqueeze(-1)
    ).clamp(FLUX_LOG10_CLIP_MIN, FLUX_LOG10_CLIP_MAX)

    isoform_mask_expanded = isoform_mask.unsqueeze(0).expand(G, R, Imax)
    log10_v = logsumexp_base10(log10_v_iso, isoform_mask_expanded)

    det_masked = detection_rate_iso * mask_float.unsqueeze(0)
    w = reliability_weights(det_masked.max(dim=-1).values, threshold=detect_threshold, steepness=detect_steepness)
    v_linear = torch.pow(10.0, log10_v)

    with torch.no_grad():
        default_v = v_linear.detach().median(dim=-2).values
    v_weighted = w * v_linear + (1.0 - w) * default_v.unsqueeze(-2)

    if use_flux_balance and S.shape[0] > 0:
        net_flux = torch.matmul(v_weighted, S.mT)
        d_max = met_degrees.max().float().clamp(min=1.0)
        degree_weight = (d_max / met_degrees.float().clamp(min=1.0)).pow(alpha_d)
        tolerance = (sigma_base * degree_weight * ref_throughput).clamp(min=1e-8)
        pyro.factor("flux_balance", dist.Normal(torch.zeros_like(net_flux), tolerance).log_prob(net_flux).sum())
    else:
        net_flux = torch.zeros((G, 0), device=device, dtype=log10_E_iso.dtype)
        tolerance = torch.zeros((G, 0), device=device, dtype=log10_E_iso.dtype)

    residual_masked = (theta_iso_kcat - theta_iso_km - theta_iso_kcat_km) * mask_float
    pyro.factor("consistency_penalty", -0.5 * consistency_weight * (residual_masked**2).sum())

    pyro.factor(
        "s_base_centering",
        dist.Normal(torch.tensor(0.0, device=device), torch.tensor(scale_anchor_sigma, device=device)).log_prob(s_base.mean()),
    )

    n_active = mask_float.sum().clamp(min=1.0)
    k_eff_mean = (kinetic_iso * mask_float).sum() / n_active
    with torch.no_grad():
        ref_k_eff_mean = ((0.5 * prior_mu_iso[:, :, idx_kcat] + 0.5 * prior_mu_iso[:, :, idx_kcat_km]) * mask_float).sum() / n_active
    pyro.factor(
        "k_eff_centering",
        dist.Normal(ref_k_eff_mean, torch.tensor(scale_anchor_sigma, device=device)).log_prob(k_eff_mean),
    )

    pyro.deterministic("log10_v_iso", log10_v_iso)
    pyro.deterministic("log10_v", log10_v)
    pyro.deterministic("net_flux", net_flux)
    pyro.deterministic("tolerance", tolerance)

    return {
        "theta_iso_km": theta_iso_km,
        "theta_iso_kcat": theta_iso_kcat,
        "theta_iso_kcat_km": theta_iso_kcat_km,
        "theta_iso_ki": theta_iso_ki,
        "omega": omega,
        "s_base": s_base,
        "sigma_base": sigma_base,
    }


def make_isokin_model_fn(
    log10_E_iso: Tensor,
    S: Tensor,
    met_degrees: Tensor,
    prior_mu_iso: Tensor,
    prior_sigma_iso: Tensor,
    isoform_mask: Tensor,
    detection_rate_iso: Tensor,
    use_flux_balance: bool = True,
    **model_kwargs,
):
    def _model():
        return isokin_stoichiometric_model(
            log10_E_iso=log10_E_iso,
            S=S,
            met_degrees=met_degrees,
            prior_mu_iso=prior_mu_iso,
            prior_sigma_iso=prior_sigma_iso,
            isoform_mask=isoform_mask,
            detection_rate_iso=detection_rate_iso,
            use_flux_balance=use_flux_balance,
            **model_kwargs,
        )

    return _model


def build_isokin_guide(model_fn, rank: int = 10):
    return AutoLowRankMultivariateNormal(model_fn, rank=rank)


class IsoKinSVITrainer:
    def __init__(
        self,
        model_fn,
        guide,
        lr: float = 1e-3,
        lr_decay: float = 0.9999,
        clip_norm: float = 10.0,
        n_iterations: int = 20000,
        convergence_window: int = 1000,
        convergence_threshold: float = 1e-3,
        clear_param_store: bool = True,
    ):
        if clear_param_store:
            pyro.clear_param_store()
        self.model_fn = model_fn
        self.guide = guide
        self.n_iterations = n_iterations
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        optimizer = ClippedAdam({"lr": lr, "lrd": lr_decay, "clip_norm": clip_norm})
        self.svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO(num_particles=1))

    def train(self) -> dict[str, Any]:
        losses: list[float] = []
        converged = False
        t0 = time.time()
        for _ in range(self.n_iterations):
            loss = float(self.svi.step())
            if not np.isfinite(loss):
                if losses:
                    break
                raise RuntimeError("Encountered non-finite ELBO loss during IsoKin SVI optimization")
            losses.append(loss)
            if self.convergence_window > 0 and len(losses) > 2 * self.convergence_window:
                recent = float(np.mean(losses[-self.convergence_window :]))
                old = float(np.mean(losses[-2 * self.convergence_window : -self.convergence_window]))
                if abs(recent - old) / (abs(old) + 1e-8) < self.convergence_threshold:
                    converged = True
                    break
        elapsed = time.time() - t0
        return {
            "losses": losses,
            "final_loss": losses[-1] if losses else float("nan"),
            "n_iterations": len(losses),
            "converged": converged,
            "elapsed_sec": elapsed,
        }

    def extract_posteriors(self, n_samples: int = 2000, return_sites: list[str] | None = None) -> dict[str, Tensor]:
        if return_sites is None:
            return_sites = [
                "theta_iso_km",
                "theta_iso_kcat",
                "theta_iso_kcat_km",
                "theta_iso_ki",
                "omega",
                "s_base",
                "sigma_base",
                "log10_v",
                "net_flux",
            ]
        pred = Predictive(self.model_fn, guide=self.guide, num_samples=n_samples, return_sites=return_sites)
        with torch.no_grad():
            return pred()


def run_isokin_svi(
    log10_E_iso: Tensor | np.ndarray,
    S: Tensor | np.ndarray,
    met_degrees: Tensor | np.ndarray,
    prior_mu_iso: Tensor | np.ndarray,
    prior_sigma_iso: Tensor | np.ndarray,
    isoform_mask: Tensor | np.ndarray,
    detection_rate_iso: Tensor | np.ndarray,
    svi_config=None,
    expression_config=None,
    seed: int = 42,
    use_flux_balance: bool = True,
) -> dict[str, Any]:
    set_all_seeds(seed)
    device_str = "cpu" if svi_config is None else svi_config.device
    device = resolve_device(device_str)

    e_t = torch.as_tensor(log10_E_iso, dtype=torch.float32, device=device)
    s_t = torch.as_tensor(S, dtype=torch.float32, device=device)
    deg_t = torch.as_tensor(met_degrees, dtype=torch.long, device=device)
    mu_t = torch.as_tensor(prior_mu_iso, dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(prior_sigma_iso, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(isoform_mask, dtype=torch.bool, device=device)
    det_t = torch.as_tensor(detection_rate_iso, dtype=torch.float32, device=device)

    model_kwargs = {}
    if expression_config is not None:
        model_kwargs["detect_threshold"] = expression_config.detect_threshold
        model_kwargs["detect_steepness"] = expression_config.detect_steepness
    if svi_config is not None:
        model_kwargs.update(
            {
                "sigma_s_global_prior": svi_config.sigma_s_global_prior,
                "sigma_s_context_prior": svi_config.sigma_s_context_prior,
                "sigma_base_prior": svi_config.sigma_base_prior,
                "consistency_weight": svi_config.consistency_weight,
                "ki_tight_sigma": svi_config.ki_tight_sigma,
                "alpha_d": svi_config.alpha_d,
                "scale_anchor_sigma": svi_config.scale_anchor_sigma,
            }
        )

    pyro.clear_param_store()
    model_fn = make_isokin_model_fn(
        log10_E_iso=e_t,
        S=s_t,
        met_degrees=deg_t,
        prior_mu_iso=mu_t,
        prior_sigma_iso=sig_t,
        isoform_mask=mask_t,
        detection_rate_iso=det_t,
        use_flux_balance=use_flux_balance,
        **model_kwargs,
    )

    latent_dim = int(4 * mu_t.shape[0] * mu_t.shape[1] + 2 * mu_t.shape[0] + 1 + e_t.shape[0] * e_t.shape[1])
    req_rank = 10 if svi_config is None else svi_config.guide_rank
    rank = max(1, min(req_rank, max(1, latent_dim - 1)))
    guide = build_isokin_guide(model_fn, rank=rank)

    trainer = IsoKinSVITrainer(
        model_fn=model_fn,
        guide=guide,
        lr=1e-3 if svi_config is None else svi_config.learning_rate,
        lr_decay=0.9999 if svi_config is None else svi_config.learning_rate_decay,
        clip_norm=10.0 if svi_config is None else svi_config.clip_norm,
        n_iterations=20000 if svi_config is None else svi_config.max_iterations,
        convergence_window=1000 if svi_config is None else svi_config.convergence_window,
        convergence_threshold=1e-3 if svi_config is None else svi_config.convergence_threshold,
        clear_param_store=False,
    )
    train_result = trainer.train()
    n_samples = 2000 if svi_config is None else svi_config.posterior_samples
    samples_torch = trainer.extract_posteriors(n_samples=n_samples)
    samples_np = {k: v.detach().cpu().numpy() for k, v in samples_torch.items()}

    return {
        "train_result": train_result,
        "posterior_samples": samples_np,
        "guide_rank": rank,
        "device": str(device),
        "flux_constrained": bool(use_flux_balance and s_t.shape[0] > 0),
    }
