from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "UniProt_ID",
    "Model_Input_Sequence",
    "SMILES",
    "pH",
    "Temperature_C",
    "Organism",
    "EnzymeType",
    "EC_Number",
    "reaction_id",
    "gene_name",
    "subsystem",
)

TASKS = ("km", "kcat", "kcat_km", "ki")
TASK_SOURCE_FAMILY = {
    "km": "single_task_km",
    "ki": "single_task_ki",
    "kcat": "multi_task",
    "kcat_km": "multi_task",
}
TASK_SOURCE_KIND = "precomputed_csv"


@dataclass(frozen=True)
class PriorTable:
    frame: pd.DataFrame

    @classmethod
    def from_precomputed_csvs(
        cls,
        mtl_csv: str | Path,
        st_km_csv: str | Path,
        st_ki_csv: str | Path,
    ) -> "PriorTable":
        frames = {
            "multi_task": _read_csv(mtl_csv),
            "single_task_km": _read_csv(st_km_csv),
            "single_task_ki": _read_csv(st_ki_csv),
        }
        rows = []
        rows.extend(_rows_for_tasks(frames["single_task_km"], ["km"], "single_task_km"))
        rows.extend(_rows_for_tasks(frames["multi_task"], ["kcat", "kcat_km"], "multi_task"))
        rows.extend(_rows_for_tasks(frames["single_task_ki"], ["ki"], "single_task_ki"))
        long_df = pd.DataFrame(rows)
        expected = [
            "reaction_id",
            "gene_name",
            "uniprot_id",
            "smiles",
            "parameter",
            "mean_log10",
            "std_log10",
            "source_checkpoint",
            "condition_pH",
            "condition_temperature",
            "organism",
            "enzyme_type",
            "source_kind",
            "prior_tier",
            "ec_number",
            "subsystem",
        ]
        return cls(long_df.loc[:, expected])

    @classmethod
    def from_artifact_registry(cls, registry) -> "PriorTable":
        return cls.from_precomputed_csvs(
            mtl_csv=registry.get("prior_mtl_csv").path,
            st_km_csv=registry.get("prior_st_km_csv").path,
            st_ki_csv=registry.get("prior_st_ki_csv").path,
        )

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def to_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)

    def match_isoform_priors(
        self,
        isoform_rows: pd.DataFrame,
        sigma_inflate: float = 1.5,
        tasks: Iterable[str] = TASKS,
    ) -> pd.DataFrame:
        return match_isoform_priors(self.frame, isoform_rows, sigma_inflate=sigma_inflate, tasks=tasks)

    def summary(self) -> dict[str, object]:
        return {
            "rows": int(len(self.frame)),
            "reactions": int(self.frame["reaction_id"].nunique()),
            "genes": int(self.frame["gene_name"].nunique()),
            "parameters": self.frame["parameter"].value_counts().sort_index().to_dict(),
            "source_checkpoints": self.frame["source_checkpoint"].value_counts().sort_index().to_dict(),
            "prior_tiers": self.frame["prior_tier"].value_counts().sort_index().to_dict(),
        }


def _read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in IDENTITY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Prior CSV {path} is missing identity columns: {missing}")
    return df


def _rows_for_tasks(df: pd.DataFrame, tasks: list[str], source_checkpoint: str) -> list[dict]:
    rows = []
    for task in tasks:
        mean_col = f"{task}_mean_log10"
        std_col = f"{task}_std_log10"
        missing = [col for col in (mean_col, std_col) if col not in df.columns]
        if missing:
            raise ValueError(f"Prior CSV for {source_checkpoint} is missing columns: {missing}")
        for _, row in df.iterrows():
            rows.append(
                {
                    "reaction_id": row["reaction_id"],
                    "gene_name": row["gene_name"],
                    "uniprot_id": row["UniProt_ID"],
                    "smiles": row["SMILES"],
                    "parameter": task,
                    "mean_log10": float(row[mean_col]),
                    "std_log10": float(row[std_col]),
                    "source_checkpoint": source_checkpoint,
                    "condition_pH": float(row["pH"]),
                    "condition_temperature": float(row["Temperature_C"]),
                    "organism": row["Organism"],
                    "enzyme_type": row["EnzymeType"],
                    "source_kind": TASK_SOURCE_KIND,
                    "prior_tier": "exact",
                    "ec_number": row["EC_Number"],
                    "subsystem": row["subsystem"],
                }
            )
    return rows


def match_isoform_priors(
    prior_frame: pd.DataFrame,
    isoform_rows: pd.DataFrame,
    sigma_inflate: float = 1.5,
    tasks: Iterable[str] = TASKS,
) -> pd.DataFrame:
    required = {"reaction_id", "gene_name"}
    missing = required - set(isoform_rows.columns)
    if missing:
        raise ValueError(f"Isoform rows missing required columns: {sorted(missing)}")

    output_rows = []
    task_list = list(tasks)
    for _, iso in isoform_rows.iterrows():
        reaction_id = iso["reaction_id"]
        gene_name = iso["gene_name"]
        for task in task_list:
            task_priors = prior_frame[prior_frame["parameter"] == task]
            task_source = TASK_SOURCE_FAMILY[task]
            exact = task_priors[
                (task_priors["reaction_id"] == reaction_id)
                & (task_priors["gene_name"].astype(str).str.upper() == str(gene_name).upper())
            ]
            if not exact.empty:
                selected = exact.iloc[0]
                tier = "exact"
                mean = float(selected["mean_log10"])
                sigma = float(selected["std_log10"])
                source = selected["source_checkpoint"]
                source_kind = selected.get("source_kind", TASK_SOURCE_KIND)
            else:
                reaction = task_priors[task_priors["reaction_id"] == reaction_id]
                if not reaction.empty:
                    tier = "reaction_average"
                    mean = float(reaction["mean_log10"].mean())
                    sigma = float(reaction["std_log10"].mean()) * sigma_inflate
                    source = task_source
                    source_kind = TASK_SOURCE_KIND
                elif not task_priors.empty:
                    tier = "global_fallback"
                    mean = float(task_priors["mean_log10"].mean())
                    sigma = float(task_priors["std_log10"].mean()) * sigma_inflate
                    source = task_source
                    source_kind = TASK_SOURCE_KIND
                else:
                    raise ValueError(f"No priors available for task {task!r}")
            output = iso.to_dict()
            output.update(
                {
                    "parameter": task,
                    "mean_log10": mean,
                    "std_log10": sigma,
                    "source_checkpoint": source,
                    "source_kind": source_kind,
                    "prior_tier": tier,
                }
            )
            output_rows.append(output)
    return pd.DataFrame(output_rows)


def get_prior_reaction_ids(mtl_csv: str | Path, st_km_csv: str | Path, st_ki_csv: str | Path) -> set[str]:
    rxn_ids: set[str] = set()
    for p in (mtl_csv, st_km_csv, st_ki_csv):
        df = pd.read_csv(p, usecols=["reaction_id"], low_memory=False)
        rxn_ids.update(df["reaction_id"].dropna().astype(str).unique())
    return rxn_ids


def _build_prediction_index(
    mtl_df: pd.DataFrame,
    st_km_df: pd.DataFrame,
    st_ki_df: pd.DataFrame,
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    index: dict[str, dict[tuple[str, str], dict[str, float]]] = {t: {} for t in TASKS}

    for _, row in mtl_df.iterrows():
        rxn_id = str(row.get("reaction_id", ""))
        gene = str(row.get("gene_name", ""))
        if not rxn_id or not gene:
            continue
        key = (rxn_id, gene)
        index["kcat"][key] = {
            "mean": float(row.get("kcat_mean_log10", 0.0)),
            "std": float(row.get("kcat_std_log10", 1.0)),
        }
        index["kcat_km"][key] = {
            "mean": float(row.get("kcat_km_mean_log10", 0.0)),
            "std": float(row.get("kcat_km_std_log10", 1.0)),
        }
        if "km_mean_log10" in mtl_df.columns and key not in index["km"]:
            index["km"][key] = {
                "mean": float(row.get("km_mean_log10", 0.0)),
                "std": float(row.get("km_std_log10", 1.0)),
            }
        if "ki_mean_log10" in mtl_df.columns and key not in index["ki"]:
            index["ki"][key] = {
                "mean": float(row.get("ki_mean_log10", 0.0)),
                "std": float(row.get("ki_std_log10", 1.0)),
            }

    for _, row in st_km_df.iterrows():
        rxn_id = str(row.get("reaction_id", ""))
        gene = str(row.get("gene_name", ""))
        if rxn_id and gene:
            index["km"][(rxn_id, gene)] = {
                "mean": float(row.get("km_mean_log10", 0.0)),
                "std": float(row.get("km_std_log10", 1.0)),
            }

    for _, row in st_ki_df.iterrows():
        rxn_id = str(row.get("reaction_id", ""))
        gene = str(row.get("gene_name", ""))
        if rxn_id and gene:
            index["ki"][(rxn_id, gene)] = {
                "mean": float(row.get("ki_mean_log10", 0.0)),
                "std": float(row.get("ki_std_log10", 1.0)),
            }

    return index


def assemble_isoform_priors_from_csvs(
    mtl_csv: str | Path,
    st_km_csv: str | Path,
    st_ki_csv: str | Path,
    reaction_ids: list[str],
    isoform_map: dict[str, Any],
    max_isoforms: int = 10,
    sigma_inflate: float = 1.5,
) -> dict[str, Any]:
    R = len(reaction_ids)
    n_isoforms = isoform_map["n_isoforms"]
    isoform_mask = isoform_map["isoform_mask"]
    isoform_rep = isoform_map["isoform_representative"]
    isoform_genes_map = isoform_map["isoform_genes"]

    mtl_df = pd.read_csv(mtl_csv, low_memory=False)
    st_km_df = pd.read_csv(st_km_csv, low_memory=False)
    st_ki_df = pd.read_csv(st_ki_csv, low_memory=False)
    pred_index = _build_prediction_index(mtl_df, st_km_df, st_ki_df)

    rxn_averages: dict[str, dict[str, dict[str, float]]] = {}
    global_fallback: dict[str, dict[str, float]] = {}
    for task in TASKS:
        rxn_averages[task] = {}
        grouped: dict[str, list[dict[str, float]]] = {}
        for (rxn_id, _), vals in pred_index[task].items():
            grouped.setdefault(rxn_id, []).append(vals)
        for rxn_id in reaction_ids:
            entries = grouped.get(rxn_id, [])
            if entries:
                rxn_averages[task][rxn_id] = {
                    "mean": float(np.mean([e["mean"] for e in entries])),
                    "std": float(np.mean([e["std"] for e in entries])),
                }
        all_means = [v["mean"] for v in pred_index[task].values()]
        all_stds = [v["std"] for v in pred_index[task].values()]
        global_fallback[task] = {
            "mean": float(np.mean(all_means)) if all_means else 0.0,
            "std": float(np.mean(all_stds)) if all_stds else 2.0,
        }

    prior_mu_iso = np.zeros((R, max_isoforms, 4), dtype=np.float64)
    prior_sigma_iso = np.ones((R, max_isoforms, 4), dtype=np.float64) * 2.0
    match_type = np.zeros((R, max_isoforms), dtype=np.int32)
    tier_names = {0: "padding", 1: "exact", 2: "reaction_average", 3: "global_fallback"}
    provenance_rows: list[dict[str, Any]] = []

    for r_idx, rxn_id in enumerate(reaction_ids):
        for i_idx in range(int(n_isoforms[r_idx])):
            if not isoform_mask[r_idx, i_idx]:
                continue
            rep_gene = isoform_rep.get((rxn_id, i_idx), "")
            iso_genes = isoform_genes_map.get((rxn_id, i_idx), [])
            iso_id = "+".join(iso_genes) if iso_genes else f"isoform_{i_idx}"

            matched = False
            per_task_tier: dict[str, str] = {}
            for t_idx, task in enumerate(TASKS):
                key = (rxn_id, rep_gene)
                if key in pred_index[task]:
                    prior_mu_iso[r_idx, i_idx, t_idx] = pred_index[task][key]["mean"]
                    prior_sigma_iso[r_idx, i_idx, t_idx] = pred_index[task][key]["std"]
                    matched = True
                    per_task_tier[task] = "exact"
                    continue

                found_gene = False
                for g in iso_genes:
                    alt_key = (rxn_id, g)
                    if alt_key in pred_index[task]:
                        prior_mu_iso[r_idx, i_idx, t_idx] = pred_index[task][alt_key]["mean"]
                        prior_sigma_iso[r_idx, i_idx, t_idx] = pred_index[task][alt_key]["std"]
                        found_gene = True
                        break
                if found_gene:
                    matched = True
                    per_task_tier[task] = "exact"
                    continue

                if rxn_id in rxn_averages[task]:
                    prior_mu_iso[r_idx, i_idx, t_idx] = rxn_averages[task][rxn_id]["mean"]
                    prior_sigma_iso[r_idx, i_idx, t_idx] = rxn_averages[task][rxn_id]["std"] * sigma_inflate
                    per_task_tier[task] = "reaction_average"
                    continue

                prior_mu_iso[r_idx, i_idx, t_idx] = global_fallback[task]["mean"]
                prior_sigma_iso[r_idx, i_idx, t_idx] = global_fallback[task]["std"] * sigma_inflate
                per_task_tier[task] = "global_fallback"

            if matched:
                any_gene = any((rxn_id, g) in pred_index[t] for g in iso_genes for t in TASKS)
                match_type[r_idx, i_idx] = 1 if any_gene else 2
            else:
                match_type[r_idx, i_idx] = 2 if rxn_id in rxn_averages["kcat"] else 3

            for t_idx, task in enumerate(TASKS):
                provenance_rows.append(
                    {
                        "reaction_id": rxn_id,
                        "isoform_idx": i_idx,
                        "isoform_id": iso_id,
                        "parameter": task,
                        "prior_mean_log10": float(prior_mu_iso[r_idx, i_idx, t_idx]),
                        "prior_std_log10": float(prior_sigma_iso[r_idx, i_idx, t_idx]),
                        "prior_tier": per_task_tier.get(task, tier_names[int(match_type[r_idx, i_idx])]),
                        "source_checkpoint": TASK_SOURCE_FAMILY[task],
                        "source_kind": TASK_SOURCE_KIND,
                    }
                )

    for r_idx in range(R):
        for i_idx in range(max_isoforms):
            if not isoform_mask[r_idx, i_idx]:
                prior_mu_iso[r_idx, i_idx, :] = 0.0
                prior_sigma_iso[r_idx, i_idx, :] = 0.0
                match_type[r_idx, i_idx] = 0

    return {
        "prior_mu_iso": prior_mu_iso,
        "prior_sigma_iso": prior_sigma_iso,
        "match_type": match_type,
        "prior_provenance": pd.DataFrame(provenance_rows),
    }
