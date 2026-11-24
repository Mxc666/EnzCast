#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEQ_COL_CANDIDATES = [
    "Sequence",
    "Protein_Sequence",
    "AA_Sequence",
    "UniProt_Sequence",
    "Model_Input_Sequence",
]


def normalize_seq(seq) -> str:
    if pd.isna(seq):
        return seq
    s = str(seq).strip().upper()
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("*", "")
    return s


def pick_sequence_column(df: pd.DataFrame) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"No sequence column found among {SEQ_COL_CANDIDATES}")


def run_mmseqs_linclust(
    unique_seqs: list[str],
    output_dir: Path,
    min_seq_id: float = 0.5,
    coverage: float = 0.8,
    cov_mode: str = "0",
    threads: int = 16,
    force_rerun: bool = False,
):
    if shutil.which("mmseqs") is None:
        warnings.warn("MMseqs2 binary not found; falling back to UniProt_ID grouping.")
        return None, None

    fasta_path = output_dir / "all_sequences.fasta"
    cluster_prefix = output_dir / "mmseqs_out"
    tmp_dir = output_dir / "mmseqs_tmp"
    cluster_tsv = output_dir / "kinbench_sequence_clusters.tsv"

    if not force_rerun and cluster_tsv.exists():
        pass
    else:
        with fasta_path.open("w", encoding="utf-8") as f:
            for idx, seq in enumerate(unique_seqs):
                f.write(f">{idx}\n{seq}\n")
        cmd = [
            "mmseqs",
            "easy-linclust",
            str(fasta_path),
            str(cluster_prefix),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "--cov-mode",
            str(cov_mode),
            "-c",
            str(coverage),
            "--threads",
            str(threads),
        ]
        try:
            subprocess.run(cmd, check=True)
            raw_tsv = Path(f"{cluster_prefix}_cluster.tsv")
            if raw_tsv.exists():
                shutil.move(str(raw_tsv), str(cluster_tsv))
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    member_to_rep = {}
    reps = set()
    with cluster_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep_s, mem_s = line.split()[:2]
            rep_id, mem_id = int(rep_s), int(mem_s)
            member_to_rep[mem_id] = rep_id
            reps.add(rep_id)
    return member_to_rep, len(reps)


def build_kinbench_splits(
    task_csvs: dict[str, Path],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_seq_id: float = 0.5,
    coverage: float = 0.8,
    threads: int = 16,
    seed: int = 42,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for task_name, csv_path in task_csvs.items():
        df = pd.read_csv(csv_path, low_memory=False)
        seq_col = pick_sequence_column(df)
        df[seq_col] = df[seq_col].apply(normalize_seq)
        df = df.dropna(subset=[seq_col]).copy()
        df["Task"] = task_name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    seq_col = pick_sequence_column(merged)
    unique_seqs = list(merged[seq_col].unique())
    seq_to_fasta_id = {s: i for i, s in enumerate(unique_seqs)}
    merged["Fasta_ID"] = merged[seq_col].map(seq_to_fasta_id)

    member_to_rep, _ = run_mmseqs_linclust(
        unique_seqs,
        output_dir,
        min_seq_id=min_seq_id,
        coverage=coverage,
        threads=threads,
    )
    if member_to_rep is not None:
        merged["Cluster_ID"] = merged["Fasta_ID"].map(lambda fid: member_to_rep.get(fid, fid))
        merged["Split_Method"] = "MMseqs2_50pct"
    else:
        uid_codes, _ = pd.factorize(merged["UniProt_ID"].astype(str))
        merged["Cluster_ID"] = uid_codes
        merged["Split_Method"] = "UniProt_Fallback"

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, holdout_idx = next(gss_outer.split(merged, groups=merged["Cluster_ID"]))
    train_df = merged.iloc[train_idx].reset_index(drop=True)
    holdout_df = merged.iloc[holdout_idx].reset_index(drop=True)

    rel_test = test_ratio / (val_ratio + test_ratio)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_idx, test_idx = next(gss_inner.split(holdout_df, groups=holdout_df["Cluster_ID"]))
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)

    assert set(train_df["Cluster_ID"]).isdisjoint(set(val_df["Cluster_ID"]))
    assert set(train_df["Cluster_ID"]).isdisjoint(set(test_df["Cluster_ID"]))
    assert set(val_df["Cluster_ID"]).isdisjoint(set(test_df["Cluster_ID"]))

    train_df.to_csv(output_dir / "kinbench_train.csv", index=False)
    val_df.to_csv(output_dir / "kinbench_val.csv", index=False)
    test_df.to_csv(output_dir / "kinbench_test.csv", index=False)
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}


def main() -> int:
    p = argparse.ArgumentParser(description="Build sequence-clustered KinBench train/val/test splits")
    p.add_argument("--kcat-csv", type=Path, required=True)
    p.add_argument("--km-csv", type=Path, required=True)
    p.add_argument("--ki-csv", type=Path, required=True)
    p.add_argument("--kcat-km-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-seq-id", type=float, default=0.5)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    counts = build_kinbench_splits(
        {
            "kcat": args.kcat_csv,
            "km": args.km_csv,
            "ki": args.ki_csv,
            "kcat_km": args.kcat_km_csv,
        },
        args.output_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        threads=args.threads,
        seed=args.seed,
    )
    print(f"KinBench splits written to {args.output_dir}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
