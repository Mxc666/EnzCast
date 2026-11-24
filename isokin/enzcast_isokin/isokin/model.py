from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enzcast_isokin.config import ExpressionConfig, SVIConfig
from enzcast_isokin.isokin.effective import (
    TASK_ORDER,
    compute_effective_kinetics_draws,
    compute_variance_decomposition,
)
from enzcast_isokin.stoichiometric_model import run_isokin_svi


@dataclass(frozen=True)
class PosteriorKinetics:
    isoform_samples: np.ndarray
    effective_mean: np.ndarray
    effective_sd: np.ndarray
    effective_q05: np.ndarray
    effective_q95: np.ndarray
    isoform_mask: np.ndarray
    tasks: tuple[str, ...]
    convergence: dict[str, object]
    prior_provenance: pd.DataFrame | None = None
    aux_samples: dict[str, np.ndarray] | None = None
    variance_decomposition_frame: pd.DataFrame | None = None

    def to_isoform_frame(
        self,
        reaction_ids: list[str] | None = None,
        isoform_genes: dict[tuple[str, int], list[str]] | None = None,
    ) -> pd.DataFrame:
        means = self.isoform_samples.mean(axis=0)
        sds = self.isoform_samples.std(axis=0)
        q05 = np.percentile(self.isoform_samples, 5, axis=0)
        q95 = np.percentile(self.isoform_samples, 95, axis=0)
        n_reactions, n_isoforms, _ = means.shape
        reaction_ids = reaction_ids or [f"reaction_{idx}" for idx in range(n_reactions)]

        prov_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
        if self.prior_provenance is not None and not self.prior_provenance.empty:
            for _, prow in self.prior_provenance.iterrows():
                prov_lookup[(str(prow["reaction_id"]), int(prow["isoform_idx"]), str(prow["parameter"]))] = prow.to_dict()

        rows = []
        for r_idx, rxn_id in enumerate(reaction_ids):
            for i_idx in range(n_isoforms):
                if not self.isoform_mask[r_idx, i_idx]:
                    continue
                genes = (isoform_genes or {}).get((rxn_id, i_idx), [])
                iso_id = "+".join(genes) if genes else f"isoform_{i_idx}"
                branch_type = "subunit_complex" if len(genes) > 1 else ("single_gene" if len(genes) == 1 else "unassigned")
                for t_idx, task in enumerate(self.tasks):
                    prov = prov_lookup.get((rxn_id, i_idx, task), {})
                    rows.append(
                        {
                            "reaction_id": rxn_id,
                            "isoform_idx": i_idx,
                            "isoform_id": iso_id,
                            "branch_type": branch_type,
                            "parameter": task,
                            "prior_mean_log10": float(prov.get("prior_mean_log10", np.nan)),
                            "prior_std_log10": float(prov.get("prior_std_log10", np.nan)),
                            "mean_log10": float(means[r_idx, i_idx, t_idx]),
                            "sd_log10": float(sds[r_idx, i_idx, t_idx]),
                            "q05_log10": float(q05[r_idx, i_idx, t_idx]),
                            "q95_log10": float(q95[r_idx, i_idx, t_idx]),
                            "prior_tier": str(prov.get("prior_tier", "exact")),
                            "source_checkpoint": str(prov.get("source_checkpoint", "NA")),
                            "is_primary": task in {"kcat", "kcat_km"},
                        }
                    )
        return pd.DataFrame(rows)

    def to_effective_frame(
        self,
        context_ids: list[str] | None = None,
        group_labels: list[tuple[str, str]] | None = None,
        reaction_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        n_contexts, n_reactions, _ = self.effective_mean.shape
        context_ids = context_ids or [f"context_{idx}" for idx in range(n_contexts)]
        reaction_ids = reaction_ids or [f"reaction_{idx}" for idx in range(n_reactions)]

        rows = []
        for g_idx, ctx_id in enumerate(context_ids):
            if group_labels is not None and g_idx < len(group_labels):
                pat, ct = group_labels[g_idx]
            elif "|" in ctx_id:
                pat, ct = ctx_id.split("|", 1)
            else:
                pat, ct = ctx_id, ""
            for r_idx, rxn_id in enumerate(reaction_ids):
                for t_idx, task in enumerate(self.tasks):
                    rows.append(
                        {
                            "context_id": ctx_id,
                            "patient_id": pat,
                            "cell_type": ct,
                            "reaction_id": rxn_id,
                            "parameter": task,
                            "mean_log10": float(self.effective_mean[g_idx, r_idx, t_idx]),
                            "sd_log10": float(self.effective_sd[g_idx, r_idx, t_idx]),
                            "q05_log10": float(self.effective_q05[g_idx, r_idx, t_idx]),
                            "q95_log10": float(self.effective_q95[g_idx, r_idx, t_idx]),
                            "is_primary": task in {"kcat", "kcat_km"},
                        }
                    )
        return pd.DataFrame(rows)

    def export_csvs(
        self,
        output_dir: str | Path,
        context_ids: list[str] | None = None,
        group_labels: list[tuple[str, str]] | None = None,
        reaction_ids: list[str] | None = None,
        isoform_genes: dict[tuple[str, int], list[str]] | None = None,
        network_summary: dict[str, Any] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        n_contexts = int(self.effective_mean.shape[0])
        n_reactions = int(self.isoform_samples.shape[1])
        context_ids = context_ids or [f"context_{idx}" for idx in range(n_contexts)]
        reaction_ids = reaction_ids or [f"reaction_{idx}" for idx in range(n_reactions)]

        isoform_path = out / "posterior_kinetics_isoform.csv"
        iso_df = self.to_isoform_frame(reaction_ids=reaction_ids, isoform_genes=isoform_genes)
        if metadata:
            iso_df = iso_df.assign(**metadata)
        iso_df.to_csv(isoform_path, index=False)
        written.append(isoform_path)

        effective_path = out / "posterior_kinetics_effective.csv"
        eff_df = self.to_effective_frame(
            context_ids=context_ids,
            group_labels=group_labels,
            reaction_ids=reaction_ids,
        )
        if metadata:
            eff_df = eff_df.assign(**metadata)
        eff_df.to_csv(effective_path, index=False)
        written.append(effective_path)

        omega_path = out / "omega_saturation.csv"
        self._omega_frame(reaction_ids=reaction_ids).to_csv(omega_path, index=False)
        written.append(omega_path)

        sbase_path = out / "s_base_substrate.csv"
        self._s_base_frame(reaction_ids=reaction_ids).to_csv(sbase_path, index=False)
        written.append(sbase_path)

        var_path = out / "variance_decomposition.csv"
        if self.variance_decomposition_frame is not None:
            self.variance_decomposition_frame.to_csv(var_path, index=False)
        else:
            pd.DataFrame(
                columns=["reaction_id", "parameter", "var_total", "var_isoform_ratio", "var_kinetic", "frac_isoform", "frac_kinetic"]
            ).to_csv(var_path, index=False)
        written.append(var_path)

        summary_path = out / "pipeline_summary.json"
        summary: dict[str, Any] = {
            "tasks": list(self.tasks),
            "n_posterior_samples": int(self.isoform_samples.shape[0]),
            "n_reactions": n_reactions,
            "n_active_isoform_branches": int(self.isoform_mask.sum()),
            "n_isoform_slots": int(self.isoform_samples.shape[2]),
            "n_contexts": n_contexts,
            "flux_constrained": bool(self.convergence.get("flux_constrained", False)),
            "convergence": {k: v for k, v in self.convergence.items() if k != "losses"},
        }
        if network_summary is not None:
            summary["network"] = network_summary
        if metadata is not None:
            summary["metadata"] = metadata
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(summary_path)
        return written

    def _omega_frame(self, reaction_ids: list[str]) -> pd.DataFrame:
        columns = ["reaction_id", "omega_mean", "omega_sd", "omega_q05", "omega_q95"]
        omega = (self.aux_samples or {}).get("omega")
        if omega is None:
            return pd.DataFrame(columns=columns)
        means = omega.mean(axis=0)
        sds = omega.std(axis=0)
        q05 = np.percentile(omega, 5, axis=0)
        q95 = np.percentile(omega, 95, axis=0)
        return pd.DataFrame(
            [
                {
                    "reaction_id": rxn_id,
                    "omega_mean": float(means[r_idx]),
                    "omega_sd": float(sds[r_idx]),
                    "omega_q05": float(q05[r_idx]),
                    "omega_q95": float(q95[r_idx]),
                }
                for r_idx, rxn_id in enumerate(reaction_ids)
            ],
            columns=columns,
        )

    def _s_base_frame(self, reaction_ids: list[str]) -> pd.DataFrame:
        columns = ["reaction_id", "s_base_mean", "s_base_sd", "s_base_q05", "s_base_q95"]
        s_base = (self.aux_samples or {}).get("s_base")
        if s_base is None:
            return pd.DataFrame(columns=columns)
        means = s_base.mean(axis=0)
        sds = s_base.std(axis=0)
        q05 = np.percentile(s_base, 5, axis=0)
        q95 = np.percentile(s_base, 95, axis=0)
        return pd.DataFrame(
            [
                {
                    "reaction_id": rxn_id,
                    "s_base_mean": float(means[r_idx]),
                    "s_base_sd": float(sds[r_idx]),
                    "s_base_q05": float(q05[r_idx]),
                    "s_base_q95": float(q95[r_idx]),
                }
                for r_idx, rxn_id in enumerate(reaction_ids)
            ],
            columns=columns,
        )


class IsoKinModel:
    def __init__(
        self,
        config: SVIConfig | None = None,
        expression_config: ExpressionConfig | None = None,
        tasks: tuple[str, ...] = TASK_ORDER,
    ) -> None:
        self.config = config or SVIConfig()
        self.expression_config = expression_config or ExpressionConfig()
        self.tasks = tuple(tasks)

    def fit(
        self,
        prior_mu: np.ndarray,
        prior_sigma: np.ndarray,
        log10_expression: np.ndarray,
        isoform_mask: np.ndarray,
        stoichiometry: np.ndarray,
        metabolite_degrees: np.ndarray | None = None,
        detection_rate: np.ndarray | None = None,
        reaction_ids: list[str] | None = None,
        prior_provenance: pd.DataFrame | None = None,
        seed: int = 42,
        use_flux_balance: bool = True,
    ) -> PosteriorKinetics:
        prior_mu_np = np.asarray(prior_mu, dtype=np.float64)
        prior_sigma_np = np.asarray(prior_sigma, dtype=np.float64)
        expr_np = np.asarray(log10_expression, dtype=np.float64)
        mask_np = np.asarray(isoform_mask, dtype=bool)
        S_np = np.asarray(stoichiometry, dtype=np.float64)

        if prior_mu_np.ndim != 3 or prior_mu_np.shape[2] != 4:
            raise ValueError(f"Expected prior_mu shape (R, Imax, 4) for [km, kcat, kcat_km, ki], got {prior_mu_np.shape}")
        if S_np.ndim != 2 or S_np.shape[1] != prior_mu_np.shape[0]:
            raise ValueError(f"Stoichiometric matrix S shape {S_np.shape} does not match R={prior_mu_np.shape[0]}")

        if metabolite_degrees is None:
            deg_np = np.count_nonzero(S_np, axis=1).astype(np.int64)
        else:
            deg_np = np.asarray(metabolite_degrees, dtype=np.int64)

        if detection_rate is None:
            det_np = np.where(mask_np[np.newaxis, :, :], 1.0, 0.0)
            det_np = np.broadcast_to(det_np, expr_np.shape).copy()
        else:
            det_np = np.asarray(detection_rate, dtype=np.float64)

        svi_out = run_isokin_svi(
            log10_E_iso=expr_np,
            S=S_np,
            met_degrees=deg_np,
            prior_mu_iso=prior_mu_np,
            prior_sigma_iso=prior_sigma_np,
            isoform_mask=mask_np,
            detection_rate_iso=det_np,
            svi_config=self.config,
            expression_config=self.expression_config,
            seed=seed,
            use_flux_balance=use_flux_balance,
        )
        samples_dict = svi_out["posterior_samples"]
        train_res = svi_out["train_result"]

        isoform_samples = np.stack([samples_dict[f"theta_iso_{t}"] for t in self.tasks], axis=-1)
        eff = compute_effective_kinetics_draws(
            posterior_samples=samples_dict,
            log10_E_iso=expr_np,
            isoform_mask=mask_np,
            tasks=self.tasks,
        )
        rxn_list = reaction_ids or [f"reaction_{idx}" for idx in range(prior_mu_np.shape[0])]
        var_df = compute_variance_decomposition(
            posterior_samples=samples_dict,
            log10_E_iso=expr_np,
            isoform_mask=mask_np,
            reaction_ids=rxn_list,
            tasks=self.tasks,
        )

        convergence = {
            "optimizer": self.config.optimizer,
            "guide": "AutoLowRankMultivariateNormal",
            "guide_rank": int(svi_out["guide_rank"]),
            "device": svi_out["device"],
            "iterations": int(train_res["n_iterations"]),
            "converged": bool(train_res["converged"]),
            "flux_constrained": bool(svi_out["flux_constrained"]),
            "n_metabolites": int(S_np.shape[0]),
            "n_reactions": int(S_np.shape[1]),
            "final_loss": float(train_res["final_loss"]),
            "elapsed_sec": float(train_res["elapsed_sec"]),
            "losses": tuple(train_res["losses"]),
        }
        aux = {
            k: samples_dict[k]
            for k in ("omega", "s_base", "sigma_base", "log10_v", "net_flux")
            if k in samples_dict
        }
        return PosteriorKinetics(
            isoform_samples=isoform_samples,
            effective_mean=eff["theta_eff_mean"],
            effective_sd=eff["theta_eff_std"],
            effective_q05=eff["theta_eff_q05"],
            effective_q95=eff["theta_eff_q95"],
            isoform_mask=mask_np,
            tasks=self.tasks,
            convergence=convergence,
            prior_provenance=prior_provenance,
            aux_samples=aux,
            variance_decomposition_frame=var_df,
        )
