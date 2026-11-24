from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from enzcast_isokin._version import __version__
from enzcast_isokin.config import PipelineConfig
from enzcast_isokin.data import AnnDataSchemaReport, validate_anndata
from enzcast_isokin.enzcast.prior_table import assemble_isoform_priors_from_csvs, get_prior_reaction_ids
from enzcast_isokin.isokin.gpr import (
    build_isoform_map,
    compute_isoform_expression,
    get_all_genes,
    load_gpr_rules,
    parse_gpr_rule,
)
from enzcast_isokin.isokin.model import IsoKinModel, PosteriorKinetics
from enzcast_isokin.network import EXCLUDED_COFACTORS, build_network, load_recon3d_stoichiometry


@dataclass(frozen=True)
class PipelineRunResult:
    schema_report: AnnDataSchemaReport
    posterior: PosteriorKinetics
    written_files: list[Path]
    summary: dict[str, object]


class KineticsPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def prepare_cohort_inputs(
        self,
        adata,
        patient_col: str,
        celltype_col: str,
        counts_layer: str | None = "auto",
    ) -> dict[str, Any]:
        recon3d_path = self.config.artifacts.get("recon3d_json").path
        gpr_path = self.config.artifacts.get("gpr_rules_json").path
        mtl_csv = self.config.artifacts.get("prior_mtl_csv").path
        km_csv = self.config.artifacts.get("prior_st_km_csv").path
        ki_csv = self.config.artifacts.get("prior_st_ki_csv").path

        for req_p in (recon3d_path, gpr_path, mtl_csv, km_csv, ki_csv):
            if not req_p.exists():
                raise FileNotFoundError(f"Required IsoKin artifact missing: {req_p}")

        gpr_rules = load_gpr_rules(gpr_path)
        all_gpr_genes: set[str] = set()
        for rule in gpr_rules.values():
            all_gpr_genes.update(get_all_genes(parse_gpr_rule(rule)))

        schema_report = validate_anndata(
            adata,
            patient_col=patient_col,
            celltype_col=celltype_col,
            gpr_genes=sorted(all_gpr_genes),
            counts_layer=counts_layer,
        )

        prior_reaction_ids = get_prior_reaction_ids(mtl_csv, km_csv, ki_csv)
        gpr_reaction_ids = set(gpr_rules.keys())
        recon3d_data = load_recon3d_stoichiometry(recon3d_path)

        candidate_rxns_initial = sorted(prior_reaction_ids | gpr_reaction_ids)
        if self.config.network.reaction_subset is not None:
            subset_set = set(self.config.network.reaction_subset)
            candidate_rxns_initial = [r for r in candidate_rxns_initial if r in subset_set]

        network_initial = build_network(
            recon3d_json_path=recon3d_path,
            common_reaction_ids=candidate_rxns_initial,
            excluded_cofactors=EXCLUDED_COFACTORS,
            min_reactions_per_metabolite=self.config.network.min_reactions_per_metabolite,
            min_metabolites_per_component=self.config.network.min_metabolites_per_component,
            recon3d_cache=recon3d_data,
        )
        network_reaction_ids = set(network_initial["reaction_ids"])

        candidate_rxns_for_iso = sorted(network_reaction_ids | prior_reaction_ids)
        if self.config.network.reaction_subset is not None:
            subset_set = set(self.config.network.reaction_subset)
            candidate_rxns_for_iso = [r for r in candidate_rxns_for_iso if r in subset_set]

        isoform_map = build_isoform_map(
            reaction_ids=candidate_rxns_for_iso,
            gpr_rules=gpr_rules,
            max_isoforms=self.config.svi.max_isoforms,
        )

        expr_data = compute_isoform_expression(
            adata=adata,
            reaction_ids=candidate_rxns_for_iso,
            isoform_map=isoform_map,
            patient_col=patient_col,
            celltype_col=celltype_col,
            counts_layer=counts_layer,
            kappa=self.config.expression.kappa,
            pseudocount=self.config.expression.pseudocount,
            temperature=self.config.expression.softmin_temperature,
            max_isoforms=self.config.svi.max_isoforms,
            target_sum_cp10k=self.config.expression.target_sum_cp10k,
        )
        expr_reaction_ids = set(expr_data["reaction_ids"])

        prior_data = assemble_isoform_priors_from_csvs(
            mtl_csv=mtl_csv,
            st_km_csv=km_csv,
            st_ki_csv=ki_csv,
            reaction_ids=candidate_rxns_for_iso,
            isoform_map=isoform_map,
            max_isoforms=self.config.svi.max_isoforms,
            sigma_inflate=self.config.svi.missing_sigma_inflate,
        )

        has_real_prior = (prior_data["match_type"] >= 1).any(axis=-1)
        prior_covered_rxns = {
            rxn_id for rxn_id, covered in zip(candidate_rxns_for_iso, has_real_prior) if covered
        }
        if not prior_covered_rxns:
            prior_covered_rxns = prior_reaction_ids

        expression_covered_rxns = gpr_reaction_ids & expr_reaction_ids
        if self.config.network.restrict_to_observed_genes:
            expression_covered_rxns &= expr_data["reactions_with_observed_gene"]

        common_rxns = sorted(prior_covered_rxns & network_reaction_ids & expression_covered_rxns)
        if not common_rxns:
            raise ValueError(
                "No common reactions remained across EnzCast priors, Recon3D internal stoichiometry, and GPR expression"
            )

        network = build_network(
            recon3d_json_path=recon3d_path,
            common_reaction_ids=common_rxns,
            excluded_cofactors=EXCLUDED_COFACTORS,
            min_reactions_per_metabolite=self.config.network.min_reactions_per_metabolite,
            min_metabolites_per_component=self.config.network.min_metabolites_per_component,
            recon3d_cache=recon3d_data,
        )
        constrained_rxn_ids: list[str] = network["reaction_ids"]
        if not constrained_rxn_ids or network["n_metabolites"] == 0:
            raise ValueError(
                "Stoichiometric network rebuild yielded 0 internal metabolites or 0 constrained reactions; "
                "ensure the input gene set spans connected metabolic reactions or lower min_metabolites_per_component."
            )

        cand_to_idx = {rxn_id: idx for idx, rxn_id in enumerate(candidate_rxns_for_iso)}
        col_idx = np.array([cand_to_idx[rxn_id] for rxn_id in constrained_rxn_ids], dtype=np.int64)

        log10_E_iso = expr_data["log10_E_iso"][:, col_idx, :]
        detection_rate_iso = expr_data["detection_rate_iso"][:, col_idx, :]
        prior_mu_iso = prior_data["prior_mu_iso"][col_idx, :, :]
        prior_sigma_iso = prior_data["prior_sigma_iso"][col_idx, :, :]
        match_type_constrained = prior_data["match_type"][col_idx, :]

        isoform_map_constrained = build_isoform_map(
            reaction_ids=constrained_rxn_ids,
            gpr_rules=gpr_rules,
            max_isoforms=self.config.svi.max_isoforms,
        )
        prov_df = prior_data["prior_provenance"]
        prov_constrained = prov_df[prov_df["reaction_id"].isin(set(constrained_rxn_ids))].reset_index(drop=True)

        return {
            "schema_report": schema_report,
            "network": network,
            "common_reaction_ids": common_rxns,
            "constrained_reaction_ids": constrained_rxn_ids,
            "group_labels": expr_data["group_labels"],
            "context_ids": [f"{p}|{c}" for p, c in expr_data["group_labels"]],
            "log10_E_iso": log10_E_iso,
            "detection_rate_iso": detection_rate_iso,
            "prior_mu_iso": prior_mu_iso,
            "prior_sigma_iso": prior_sigma_iso,
            "match_type": match_type_constrained,
            "isoform_map": isoform_map_constrained,
            "prior_provenance": prov_constrained,
        }

    def run_h5ad(
        self,
        input_h5ad: str | Path,
        patient_col: str,
        celltype_col: str,
        output_dir: str | Path,
        counts_layer: str | None = "auto",
        seed: int = 42,
    ) -> PipelineRunResult:
        import anndata as ad

        input_path = Path(input_h5ad).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        adata = ad.read_h5ad(input_path)

        prepared = self.prepare_cohort_inputs(
            adata=adata,
            patient_col=patient_col,
            celltype_col=celltype_col,
            counts_layer=counts_layer,
        )
        schema_report: AnnDataSchemaReport = prepared["schema_report"]
        network = prepared["network"]
        isoform_map = prepared["isoform_map"]
        constrained_rxn_ids = prepared["constrained_reaction_ids"]
        tasks = tuple(self.config.posterior_tasks)

        posterior = IsoKinModel(
            config=self.config.svi,
            expression_config=self.config.expression,
            tasks=tasks,
        ).fit(
            prior_mu=prepared["prior_mu_iso"],
            prior_sigma=prepared["prior_sigma_iso"],
            log10_expression=prepared["log10_E_iso"],
            isoform_mask=isoform_map["isoform_mask"],
            stoichiometry=network["S"],
            metabolite_degrees=network["metabolite_degrees"],
            detection_rate=prepared["detection_rate_iso"],
            reaction_ids=constrained_rxn_ids,
            prior_provenance=prepared["prior_provenance"],
            seed=seed,
            use_flux_balance=True,
        )

        network_summary = {
            "n_common_reactions": len(prepared["common_reaction_ids"]),
            "n_constrained_reactions": int(network["n_reactions"]),
            "n_internal_metabolites": int(network["n_metabolites"]),
            "n_connected_components": len(network["components"]),
            "n_multi_isoform_reactions": int(isoform_map["n_multi_iso"]),
            "n_subunit_complex_branches": int(isoform_map["n_complex_branches"]),
            "stoichiometric_matrix_shape": [int(network["S"].shape[0]), int(network["S"].shape[1])],
        }
        metadata = self._metadata(
            input_path=input_path,
            schema_report=schema_report,
            reaction_ids=constrained_rxn_ids,
            posterior=posterior,
        )
        written_files = posterior.export_csvs(
            output_path,
            context_ids=prepared["context_ids"],
            group_labels=prepared["group_labels"],
            reaction_ids=constrained_rxn_ids,
            isoform_genes=isoform_map["isoform_genes"],
            network_summary=network_summary,
            metadata=metadata,
        )
        summary = {
            "input_file": str(input_path),
            "output_dir": str(output_path),
            "flux_constrained": bool(posterior.convergence.get("flux_constrained", False)),
            "retained_contexts": prepared["context_ids"],
            "retained_reactions": constrained_rxn_ids,
            "tasks": list(tasks),
            "primary_tasks": list(self.config.primary_posterior_tasks),
            "network": network_summary,
            "written_files": [str(path) for path in written_files],
            "metadata": metadata,
        }
        return PipelineRunResult(
            schema_report=schema_report,
            posterior=posterior,
            written_files=written_files,
            summary=summary,
        )

    def _metadata(
        self,
        input_path: Path,
        schema_report: AnnDataSchemaReport,
        reaction_ids: list[str],
        posterior: PosteriorKinetics,
    ) -> dict[str, object]:
        return {
            "package_version": __version__,
            "config_hash": _config_hash(self.config),
            "artifact_ids": ",".join(sorted(self.config.artifacts.as_dict())),
            "input_file": str(input_path.name),
            "retained_context_count": int(schema_report.n_contexts),
            "retained_reaction_count": int(len(reaction_ids)),
            "flux_constrained": bool(posterior.convergence.get("flux_constrained", False)),
            "svi_converged": bool(posterior.convergence.get("converged", False)),
            "svi_iterations": int(posterior.convergence.get("iterations", 0)),
        }


def _config_hash(config: PipelineConfig) -> str:
    payload = {
        "default_condition": asdict(config.default_condition),
        "network": asdict(config.network),
        "expression": asdict(config.expression),
        "svi": asdict(config.svi),
        "posterior_tasks": config.posterior_tasks,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
