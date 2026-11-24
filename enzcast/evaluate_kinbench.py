#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from enzcast.dataset import TASK_NAMES
from enzcast.predict import DEFAULT_ESM_CONFIG_DIR, run_prediction

LEVELS = [0.50, 0.80, 0.90, 0.95]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def gaussian_nll(y, mu, var) -> np.ndarray:
    y, mu, var = np.asarray(y, dtype=float), np.asarray(mu, dtype=float), np.asarray(var, dtype=float)
    return 0.5 * (np.log(2.0 * np.pi) + np.log(var) + ((y - mu) ** 2) / var)


def point_metrics(y, mu) -> dict[str, float]:
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    n = int(len(y))
    if n == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "spearman": np.nan}
    err = mu - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    pearson = float(pearsonr(y, mu).statistic) if n > 1 and np.std(y) > 0 and np.std(mu) > 0 else np.nan
    spearman = float(spearmanr(y, mu).statistic) if n > 1 and np.std(y) > 0 and np.std(mu) > 0 else np.nan
    return {
        "n": n,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson,
        "spearman": spearman,
    }


def distribution_metrics(y, mu, var) -> dict[str, float]:
    ans = point_metrics(y, mu)
    if ans["n"] == 0:
        ans["nll"] = np.nan
    else:
        ans["nll"] = float(np.mean(gaussian_nll(y, mu, var)))
    for level in LEVELS:
        z = norm.ppf((1.0 + level) / 2.0)
        half = z * np.sqrt(np.asarray(var, dtype=float))
        tag = int(level * 100)
        ans[f"coverage_{tag}"] = float(np.mean(np.abs(np.asarray(y) - np.asarray(mu)) <= half)) if ans["n"] else np.nan
        ans[f"width_{tag}"] = float(np.mean(2.0 * half)) if ans["n"] else np.nan
    return ans


def evaluate_predictions_frame(df: pd.DataFrame, scaling_factors: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for task in TASK_NAMES:
        sub = df[df["Task"].astype(str).str.lower() == task].copy()
        mu_col = f"{task}_mean_log10"
        var_col = f"{task}_var_log10"
        if sub.empty or mu_col not in sub.columns or var_col not in sub.columns:
            continue
        ok = sub["Log10_Value"].notna() & sub[mu_col].notna() & sub[var_col].notna()
        use = sub[ok]
        if use.empty:
            continue

        raw_m = distribution_metrics(use["Log10_Value"].values, use[mu_col].values, use[var_col].values)
        raw_m["task"] = task
        raw_m["calibration"] = "raw"
        raw_m["variance_scale_a"] = 1.0
        rows.append(raw_m)

        if scaling_factors and task in scaling_factors:
            a = float(scaling_factors[task])
            scaled_m = distribution_metrics(use["Log10_Value"].values, use[mu_col].values, use[var_col].values * a)
            scaled_m["task"] = task
            scaled_m["calibration"] = "val_scaled"
            scaled_m["variance_scale_a"] = a
            rows.append(scaled_m)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    default_kb = CODE_ROOT.parent / "Dataset" / "kinbench"
    default_model_dir = CODE_ROOT.parent / "Model" / "benchmark"
    p = argparse.ArgumentParser(description="Evaluate EnzCast benchmark checkpoints on KinBench splits")
    p.add_argument("--split-csv", type=Path, default=default_kb / "kinbench_test.csv")
    p.add_argument("--predictions-csv", type=Path, default=None, help="Evaluate pre-exported prediction CSV without re-running model")
    p.add_argument("--protein-graph-dir", type=Path, default=None)
    p.add_argument("--substrate-graph-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=default_model_dir)
    p.add_argument("--variance-scaling-json", type=Path, default=default_model_dir / "variance_scaling_val.json")
    p.add_argument("--esm-config-dir", type=Path, default=DEFAULT_ESM_CONFIG_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_csv is not None:
        pred_df = pd.read_csv(args.predictions_csv, low_memory=False)
    else:
        if args.protein_graph_dir is None or args.substrate_graph_dir is None:
            raise ValueError("--protein-graph-dir and --substrate-graph-dir are required when running model inference")
        df = pd.read_csv(args.split_csv, low_memory=False)
        if args.limit is not None:
            df = df.iloc[: args.limit].copy()
        pred_df = run_prediction(
            df=df,
            suite="benchmark",
            tasks=list(TASK_NAMES),
            model_dir=args.model_dir,
            esm_config_dir=args.esm_config_dir,
            protein_graph_dir=args.protein_graph_dir,
            substrate_graph_dir=args.substrate_graph_dir,
            device_str=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_autocast=False,
        )
        pred_df.to_csv(args.output_dir / "kinbench_predictions.csv", index=False)

    scaling = None
    if args.variance_scaling_json is not None and args.variance_scaling_json.exists():
        raw_json = json.loads(args.variance_scaling_json.read_text(encoding="utf-8"))
        if "tasks" in raw_json:
            scaling = {k: float(v["a"]) for k, v in raw_json["tasks"].items()}

    metrics_df = evaluate_predictions_frame(pred_df, scaling_factors=scaling)
    metrics_path = args.output_dir / "kinbench_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df.to_string(index=False))
    print(f"Saved evaluation metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
