import json

import anndata as ad
import numpy as np
import pandas as pd

from test_pipeline import make_h5ad, make_project_root


def test_validate_h5ad_cli_reports_missing_columns(tmp_path, capsys):
    from enzcast_isokin.cli import main

    bad_h5ad = tmp_path / "bad.h5ad"
    obs = pd.DataFrame({"sample_id": ["P1"]}, index=["cell1"])
    var = pd.DataFrame(index=["GENE1"])
    ad.AnnData(X=np.ones((1, 1)), obs=obs, var=var, layers={"counts": np.ones((1, 1))}).write_h5ad(bad_h5ad)

    code = main(["validate-h5ad", "--input", str(bad_h5ad), "--patient-col", "patient_id", "--celltype-col", "cell_type"])

    captured = capsys.readouterr()
    assert code == 1
    assert "missing required obs columns" in captured.err


def test_infer_and_summarize_cli_runs_flux_constrained_fitting(tmp_path, capsys):
    from enzcast_isokin.cli import main

    config = make_project_root(tmp_path / "project")
    input_h5ad = tmp_path / "cohort.h5ad"
    output_dir = tmp_path / "out"
    make_h5ad(input_h5ad)

    infer_code = main(
        [
            "infer",
            "--input",
            str(input_h5ad),
            "--patient-col",
            "patient_id",
            "--celltype-col",
            "cell_type",
            "--project-root",
            str(config.project_root),
            "--out",
            str(output_dir),
            "--observed-genes-only",
            "--min-component-metabolites",
            "2",
            "--max-iterations",
            "2",
            "--posterior-samples",
            "3",
            "--seed",
            "13",
        ]
    )

    assert infer_code == 0
    summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["flux_constrained"] is True
    assert summary["network"]["n_internal_metabolites"] == 3
    assert summary["network"]["n_subunit_complex_branches"] == 1

    effective = pd.read_csv(output_dir / "posterior_kinetics_effective.csv")
    assert {"package_version", "config_hash", "flux_constrained", "input_file"} <= set(effective.columns)
    assert set(effective["parameter"]) == {"km", "kcat", "kcat_km", "ki"}

    summarize_code = main(["summarize", "--input", str(output_dir)])
    captured = capsys.readouterr()

    assert summarize_code == 0
    assert "kcat_km" in captured.out
    assert "posterior_kinetics_effective.csv" in captured.out
