#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
RELEASE_ROOT = CODE_ROOT.parent

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "isokin"))

from enzcast.evaluate_kinbench import evaluate_predictions_frame
from enzcast.featurize import pdb_to_protein_graph, smiles_to_graph
from enzcast.predict import DEFAULT_ESM_CONFIG_DIR, run_prediction
from enzcast.train import main as run_enzcast_train
from enzcast_isokin.cli import main as run_isokin_cli
from kinbench.verify_splits import verify_kinbench_directory


def run_all_checks(device: str = "cpu") -> dict:
    results = {}

    kb_dir = RELEASE_ROOT / "Dataset" / "kinbench"
    kb_report = verify_kinbench_directory(kb_dir, strict_counts=True)
    manifest_df = pd.read_csv(kb_dir / "kinbench_structure_manifest.csv")
    pdb_tar = kb_dir / "structures" / "protein_structures_pdb.tar.gz"
    prot_tar = kb_dir / "features" / "protein_graphs.tar"
    sub_tar = kb_dir / "features" / "substrate_graphs.tar"

    sample_pdb = next(iter(sorted((HERE / "structures").glob("*.pdb"))))
    sample_graph = pdb_to_protein_graph(sample_pdb)
    sample_mol = smiles_to_graph("CCO")
    if sample_graph is None or sample_mol is None:
        raise AssertionError("Failed to featurize sample PDB or SMILES")

    results["kinbench"] = {
        "total_records": kb_report["total_records"],
        "cluster_disjoint": kb_report["cluster_disjoint"],
        "splits": {k: v["rows"] for k, v in kb_report["splits"].items()},
        "structure_manifest_count": int(len(manifest_df)),
        "pdb_archive_exists": pdb_tar.exists(),
        "protein_graphs_tar_exists": prot_tar.exists(),
        "substrate_graphs_tar_exists": sub_tar.exists(),
        "sample_pdb_residues": int(sample_graph.x.shape[0]),
    }

    sample_csv = HERE / "kinbench_sample.csv"
    ref_csv = HERE / "kinbench_sample_reference_predictions.csv"
    prot_dir = HERE / "features" / "protein_graphs"
    sub_dir = HERE / "features" / "substrate_graphs"
    pdb_dir = HERE / "structures"

    sample_df = pd.read_csv(sample_csv)
    ref_df = pd.read_csv(ref_csv)

    with tempfile.TemporaryDirectory(prefix="enzcast_train_verify_") as tmp_train:
        train_out = Path(tmp_train)
        rc = run_enzcast_train(
            [
                "--mode",
                "multitask",
                "--data-mode",
                "benchmark",
                "--train-csv",
                str(sample_csv),
                "--val-csv",
                str(sample_csv),
                "--protein-graph-dir",
                str(prot_dir),
                "--substrate-graph-dir",
                str(sub_dir),
                "--esm-model",
                str(DEFAULT_ESM_CONFIG_DIR),
                "--output-dir",
                str(train_out),
                "--pretrain-epochs",
                "0",
                "--stage1-epochs",
                "1",
                "--max-steps",
                "2",
                "--grad-accum-steps",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
                "--device",
                device,
            ]
        )
        if rc != 0:
            raise AssertionError(f"train.py exited with status {rc}")

        saved_ckpt = train_out / "enzcast_multitask.pt"
        saved_meta = train_out / "enzcast_multitask_metadata.json"
        if not saved_ckpt.exists() or not saved_meta.exists():
            raise AssertionError("train.py did not emit expected checkpoint and metadata files")

        reloaded_pred = run_prediction(
            df=sample_df.iloc[:2].copy(),
            suite="benchmark",
            tasks=["km", "kcat", "kcat_km", "ki"],
            model_dir=train_out,
            metadata_path=saved_meta,
            checkpoint_override=saved_ckpt,
            model_key_override="multitask",
            protein_graph_dir=prot_dir,
            substrate_graph_dir=sub_dir,
            device_str=device,
            batch_size=2,
            num_workers=0,
            use_autocast=False,
        )
        pred_arr = reloaded_pred[["km_mean_log10", "kcat_mean_log10", "kcat_km_mean_log10", "ki_mean_log10"]].to_numpy()
        if not np.isfinite(pred_arr).all():
            raise AssertionError("Reloaded trained checkpoint produced non-finite predictions")
        saved_meta_json = json.loads(saved_meta.read_text(encoding="utf-8"))

    bench_pred = run_prediction(
        df=sample_df,
        suite="benchmark",
        tasks=["km", "kcat", "kcat_km", "ki"],
        protein_graph_dir=prot_dir,
        substrate_graph_dir=sub_dir,
        device_str=device,
        batch_size=4,
        num_workers=0,
        use_autocast=False,
    )

    max_abs_mu_diff = 0.0
    max_abs_var_diff = 0.0
    for idx, r_row in ref_df.iterrows():
        task = str(r_row["task"])
        mu_got = float(bench_pred.loc[idx, f"{task}_mean_log10"])
        var_got = float(bench_pred.loc[idx, f"{task}_var_log10"])
        max_abs_mu_diff = max(max_abs_mu_diff, abs(mu_got - float(r_row["mu_log10"])))
        max_abs_var_diff = max(max_abs_var_diff, abs(var_got - float(r_row["var_log10"])))

    if max_abs_mu_diff > 1e-4 or max_abs_var_diff > 1e-4:
        raise AssertionError(f"Benchmark prediction drift exceeded tolerance: mu={max_abs_mu_diff}, var={max_abs_var_diff}")

    val_scale_json = json.loads((RELEASE_ROOT / "Model" / "benchmark" / "variance_scaling_val.json").read_text(encoding="utf-8"))
    scaling = {k: float(v["a"]) for k, v in val_scale_json["tasks"].items()}
    eval_df = evaluate_predictions_frame(bench_pred, scaling_factors=scaling)

    new_pair_df = sample_df.iloc[:2].copy()
    fd_pred = run_prediction(
        df=new_pair_df,
        suite="fulldata",
        tasks=["km", "kcat", "kcat_km", "ki"],
        protein_graph_dir=None,
        substrate_graph_dir=None,
        pdb_dir=pdb_dir,
        device_str=device,
        batch_size=2,
        num_workers=0,
        use_autocast=False,
    )
    if not np.isfinite(fd_pred[["km_mean_log10", "kcat_mean_log10", "kcat_km_mean_log10", "ki_mean_log10"]].to_numpy()).all():
        raise AssertionError("Full-data prediction from raw PDB + SMILES produced non-finite values")

    results["enzcast"] = {
        "train_smoke_optimizer_steps": int(saved_meta_json["optimizer_steps_completed"]),
        "train_checkpoint_reloaded_rows": int(len(reloaded_pred)),
        "sample_rows": int(len(sample_df)),
        "benchmark_max_abs_mu_diff_vs_official": max_abs_mu_diff,
        "benchmark_max_abs_var_diff_vs_official": max_abs_var_diff,
        "evaluated_metric_rows": int(len(eval_df)),
        "fulldata_pdb_on_the_fly_rows": int(len(fd_pred)),
    }

    with tempfile.TemporaryDirectory(prefix="isokin_verify_") as tmp:
        out_dir = Path(tmp) / "isokin_out"
        sample_h5ad = HERE / "sample_cohort.h5ad"
        cli_rc = run_isokin_cli(
            [
                "infer",
                "--input",
                str(sample_h5ad),
                "--patient-col",
                "subject_id",
                "--celltype-col",
                "cell_state",
                "--project-root",
                str(RELEASE_ROOT),
                "--out",
                str(out_dir),
                "--observed-genes-only",
                "--max-iterations",
                "10",
                "--posterior-samples",
                "16",
                "--device",
                device,
                "--seed",
                "42",
            ]
        )
        if cli_rc != 0:
            raise AssertionError(f"enzcast-isokin infer exited with status {cli_rc}")

        summary = json.loads((out_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
        eff_df = pd.read_csv(out_dir / "posterior_kinetics_effective.csv")
        iso_df = pd.read_csv(out_dir / "posterior_kinetics_isoform.csv")
        omega_df = pd.read_csv(out_dir / "omega_saturation.csv")
        sbase_df = pd.read_csv(out_dir / "s_base_substrate.csv")

        if not summary["flux_constrained"]:
            raise AssertionError("IsoKin ran without stoichiometric flux constraint")
        if summary["network"]["n_internal_metabolites"] <= 0 or summary["network"]["n_subunit_complex_branches"] <= 0:
            raise AssertionError(f"Unexpected network/GPR structure: {summary['network']}")
        if not np.isfinite(eff_df["mean_log10"]).all() or not np.isfinite(iso_df["mean_log10"]).all():
            raise AssertionError("Non-finite IsoKin posterior values")

        results["isokin"] = {
            "flux_constrained": bool(summary["flux_constrained"]),
            "n_contexts": int(summary["n_contexts"]),
            "n_constrained_reactions": int(summary["network"]["n_constrained_reactions"]),
            "n_internal_metabolites": int(summary["network"]["n_internal_metabolites"]),
            "n_multi_isoform_reactions": int(summary["network"]["n_multi_isoform_reactions"]),
            "n_subunit_complex_branches": int(summary["network"]["n_subunit_complex_branches"]),
            "effective_rows": int(len(eff_df)),
            "isoform_rows": int(len(iso_df)),
            "omega_rows": int(len(omega_df)),
            "s_base_rows": int(len(sbase_df)),
            "svi_final_loss": float(summary["convergence"]["final_loss"]),
        }

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal end-to-end verification of EnzCast-IsoKin workflows")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    res = run_all_checks(device=args.device)
    text = json.dumps(res, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
