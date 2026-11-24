from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from enzcast_isokin.isokin.gpr import build_gene_index


@dataclass(frozen=True)
class AnnDataSchemaReport:
    n_cells: int
    n_genes: int
    n_contexts: int
    group_key: tuple[str, str]
    retained_genes: tuple[str, ...]
    missing_gpr_genes: tuple[str, ...]
    counts_layer: str | None
    context_labels: tuple[str, ...]
    retained_cell_ids: tuple[str, ...]
    excluded_cell_ids: tuple[str, ...] = ()
    excluded_contexts: tuple[str, ...] = ()


def validate_anndata(
    adata,
    patient_col: str,
    celltype_col: str,
    gpr_genes: Iterable[str] | None = None,
    counts_layer: str | None = "auto",
) -> AnnDataSchemaReport:
    if not hasattr(adata, "obs") or not hasattr(adata, "var"):
        raise TypeError("Expected an AnnData-like object with .obs and .var attributes")

    missing_obs = [col for col in (patient_col, celltype_col) if col not in adata.obs.columns]
    if missing_obs:
        raise ValueError(f"AnnData is missing required obs columns: {', '.join(missing_obs)}")

    if adata.n_obs == 0:
        raise ValueError("AnnData contains no cells")
    if adata.n_vars == 0:
        raise ValueError("AnnData contains no genes")

    selected_counts_layer: str | None = None
    if counts_layer == "auto":
        selected_counts_layer = "counts" if "counts" in adata.layers else None
    elif counts_layer not in (None, "X_log1p"):
        if counts_layer not in adata.layers:
            raise ValueError(f"AnnData is missing requested counts layer: {counts_layer}")
        selected_counts_layer = counts_layer

    obs = adata.obs[[patient_col, celltype_col]]
    missing_label_mask = obs.isna().any(axis=1) | obs.apply(lambda col: col.astype(str).str.strip().eq("")).any(axis=1)
    if missing_label_mask.any():
        bad_cells = ", ".join(str(idx) for idx in obs.index[missing_label_mask][:10])
        raise ValueError(f"AnnData has cells with missing patient/cell-type labels: {bad_cells}")

    if obs.empty:
        raise ValueError("No cells remain after dropping missing patient/cell-type labels")

    context_labels = tuple(
        sorted(f"{patient}|{celltype}" for patient, celltype in obs.drop_duplicates().itertuples(index=False))
    )
    gene_index = build_gene_index(adata)
    var_symbols = tuple(gene_index.keys()) if gene_index else tuple(str(name) for name in adata.var_names)
    available_upper = {gene.upper(): gene for gene in var_symbols}

    if gpr_genes is None:
        retained_genes = var_symbols
        missing_gpr_genes: tuple[str, ...] = ()
    else:
        retained = []
        missing = []
        for gene in gpr_genes:
            key = str(gene).upper()
            if key in available_upper:
                retained.append(available_upper[key])
            else:
                missing.append(str(gene))
        retained_genes = tuple(retained)
        missing_gpr_genes = tuple(missing)
        if not retained_genes:
            visible = ", ".join(var_symbols[:10])
            requested = ", ".join(missing_gpr_genes[:10])
            raise ValueError(
                "No requested GPR genes were found in AnnData gene symbols. "
                f"Requested examples: {requested}. Visible gene examples: {visible}"
            )

    return AnnDataSchemaReport(
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_contexts=len(context_labels),
        group_key=(patient_col, celltype_col),
        retained_genes=retained_genes,
        missing_gpr_genes=missing_gpr_genes,
        counts_layer=selected_counts_layer,
        context_labels=context_labels,
        retained_cell_ids=tuple(str(idx) for idx in obs.index),
        excluded_cell_ids=(),
    )
