#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "EC_Number",
    "Organism",
    "Substrate",
    "SMILES",
    "UniProt_ID",
    "Sequence",
    "Value",
    "Log10_Value",
    "Unit",
    "pH",
    "Temperature_C",
    "EnzymeType",
    "Source_Database",
    "Is_Predicted",
    "Sequence_raw",
    "Model_Input_Sequence",
    "Crop_Start",
    "Crop_End",
    "Task",
    "Fasta_ID",
    "Cluster_ID",
    "Split_Method",
]

OFFICIAL_COUNTS = {"train": 285892, "val": 36770, "test": 36990}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_kinbench_directory(kinbench_dir: Path, strict_counts: bool = True) -> dict:
    report = {"splits": {}, "cluster_disjoint": False, "total_records": 0}
    cluster_sets = {}

    for split in ["train", "val", "test"]:
        csv_path = kinbench_dir / f"kinbench_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        cols = list(df.columns)
        if cols != EXPECTED_COLUMNS:
            raise ValueError(f"Column mismatch in {csv_path.name}: {cols}")
        if strict_counts and len(df) != OFFICIAL_COUNTS[split]:
            raise ValueError(f"Row count mismatch in {csv_path.name}: got {len(df)}, expected {OFFICIAL_COUNTS[split]}")

        cluster_sets[split] = set(df["Cluster_ID"].dropna().astype(int).tolist())
        task_counts = {k: int(v) for k, v in df["Task"].value_counts().to_dict().items()}
        report["splits"][split] = {
            "file": str(csv_path),
            "rows": int(len(df)),
            "unique_clusters": len(cluster_sets[split]),
            "task_counts": task_counts,
            "sha256": sha256_file(csv_path),
        }
        report["total_records"] += int(len(df))

    disjoint = (
        cluster_sets["train"].isdisjoint(cluster_sets["val"])
        and cluster_sets["train"].isdisjoint(cluster_sets["test"])
        and cluster_sets["val"].isdisjoint(cluster_sets["test"])
    )
    if not disjoint:
        raise ValueError("Sequence cluster leakage detected across KinBench splits!")
    report["cluster_disjoint"] = True
    return report


def main() -> int:
    default_dir = Path(__file__).resolve().parents[2] / "Dataset" / "kinbench"
    p = argparse.ArgumentParser(description="Verify KinBench split schema, counts, and cluster disjointness")
    p.add_argument("--kinbench-dir", type=Path, default=default_dir)
    p.add_argument("--no-strict-counts", action="store_true")
    args = p.parse_args()

    report = verify_kinbench_directory(args.kinbench_dir, strict_counts=not args.no_strict_counts)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
