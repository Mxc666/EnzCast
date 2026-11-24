import anndata as ad
import numpy as np
import pandas as pd
import pytest

from enzcast_isokin.artifacts import ArtifactRegistry
from enzcast_isokin.data import validate_anndata


def make_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "cell_type": ["Tumor", "Normal", "Tumor", "Normal"],
            "disease_state": ["CRC", "Adjacent", "CRC", "Adjacent"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    var = pd.DataFrame(index=["GENE1", "GENE2", "MISSING"])
    layers = {"counts": np.array([[1, 0, 0], [2, 1, 0], [0, 3, 0], [1, 1, 0]], dtype=float)}
    return ad.AnnData(X=np.log1p(layers["counts"]), obs=obs, var=var, layers=layers)


def test_validate_anndata_reports_contexts_genes_and_counts_layer():
    adata = make_adata()

    report = validate_anndata(
        adata,
        patient_col="patient_id",
        celltype_col="cell_type",
        gpr_genes=["GENE1", "GENE2", "GENE3"],
        counts_layer="counts",
    )

    assert report.n_cells == 4
    assert report.n_genes == 3
    assert report.n_contexts == 4
    assert report.group_key == ("patient_id", "cell_type")
    assert report.retained_genes == ("GENE1", "GENE2")
    assert report.missing_gpr_genes == ("GENE3",)
    assert report.counts_layer == "counts"
    assert report.context_labels == ("P1|Normal", "P1|Tumor", "P2|Normal", "P2|Tumor")
    assert report.retained_cell_ids == ("cell1", "cell2", "cell3", "cell4")
    assert report.excluded_cell_ids == ()


def test_validate_anndata_fails_with_actionable_missing_columns():
    adata = make_adata()

    with pytest.raises(ValueError, match="missing required obs columns.*sample_id"):
        validate_anndata(adata, patient_col="sample_id", celltype_col="cell_type")


def test_validate_anndata_fails_on_missing_context_labels():
    adata = make_adata()
    adata.obs.loc["cell2", "patient_id"] = np.nan

    with pytest.raises(ValueError, match="missing patient/cell-type labels.*cell2"):
        validate_anndata(adata, patient_col="patient_id", celltype_col="cell_type")


def test_validate_anndata_fails_on_blank_context_labels():
    adata = make_adata()
    adata.obs.loc["cell3", "cell_type"] = "   "

    with pytest.raises(ValueError, match="missing patient/cell-type labels.*cell3"):
        validate_anndata(adata, patient_col="patient_id", celltype_col="cell_type")


def test_validate_anndata_fails_when_no_gpr_genes_are_retained():
    adata = make_adata()

    with pytest.raises(ValueError, match="No requested GPR genes were found.*GENE9"):
        validate_anndata(
            adata,
            patient_col="patient_id",
            celltype_col="cell_type",
            gpr_genes=["GENE9", "GENE10"],
        )


def test_artifact_manifest_is_deterministic_and_local_only(tmp_path):
    registry = ArtifactRegistry.default(project_root=tmp_path)
    existing = registry.get("prior_mtl_csv").path
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("header\n", encoding="utf-8")

    manifest = registry.manifest()
    row_by_key = {row["key"]: row for row in manifest}

    assert [row["key"] for row in manifest] == sorted(row_by_key)
    assert row_by_key["prior_mtl_csv"]["exists"] is True
    assert row_by_key["prior_st_km_csv"]["exists"] is False
    assert row_by_key["recon3d_json"]["path"] == str(tmp_path / "Dataset" / "isokin" / "recon3d_model.json")
    assert all(row["remote_uri"] is None for row in manifest)
    assert all(row["download_attempted"] is False for row in manifest)
    assert row_by_key["checkpoint_mtl"]["path"] == str(
        tmp_path / "Model" / "fulldata_priors" / "enzcast_fulldata_multitask.pt"
    )
