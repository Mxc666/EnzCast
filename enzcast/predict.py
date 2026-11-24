#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from enzcast.dataset import (
    TASK_IDX,
    TASK_NAMES,
    TASK_TO_HEAD,
    EnzCastInferenceDataset,
    collate_fn_inference,
    inverse_gaussian,
    load_suite_metadata,
)
from enzcast.model import EnzCastModel, detect_gpu_config

DEFAULT_ESM_CONFIG_DIR = HERE / "assets" / "esm2_t33_650M_UR50D"
DEFAULT_MODEL_ROOT = CODE_ROOT.parent / "Model"

SUITE_DEFAULTS = {
    "benchmark": {
        "subdir": "benchmark",
        "metadata": "enzcast_benchmark_metadata.json",
    },
    "fulldata": {
        "subdir": "fulldata_priors",
        "metadata": "enzcast_fulldata_metadata.json",
    },
}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_single_checkpoint(
    df: pd.DataFrame,
    checkpoint_path: Path,
    metadata: dict,
    model_key: str,
    target_tasks: list[str],
    esm_config_dir: Path,
    device: torch.device,
    protein_graph_dir: Path | None,
    substrate_graph_dir: Path | None,
    pdb_dir: Path | None,
    batch_size: int,
    num_workers: int,
    use_autocast: bool = False,
    mask_mode: str = "observed",
) -> dict[int, dict[str, float]]:
    arch = metadata["model_architecture"]
    tokenizer = AutoTokenizer.from_pretrained(str(esm_config_dir))
    ds = EnzCastInferenceDataset(
        dataframe=df,
        tokenizer=tokenizer,
        metadata=metadata,
        model_key=model_key,
        protein_graph_dir=protein_graph_dir,
        substrate_graph_dir=substrate_graph_dir,
        pdb_dir=pdb_dir,
        mask_mode=mask_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    from_cfg_only = not (
        (esm_config_dir / "model.safetensors").exists()
        or (esm_config_dir / "pytorch_model.bin").exists()
    )
    model = EnzCastModel(
        str(esm_config_dir),
        num_species=int(arch["num_embedding_organisms"]),
        num_types=int(arch["num_types"]),
        num_ecs=int(arch["num_ecs"]),
        recon_dim=int(arch["recon_dim"]),
        ec_init_matrix=None,
        from_config_only=from_cfg_only,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_autocast and device.type == "cuda")
        else nullcontext()
    )

    stats = metadata["stats_int"]
    predictions_by_row: dict[int, dict[str, float]] = {}

    with torch.no_grad():
        for ids, mask, struct, sub, conds, orig_indices in tqdm(
            loader,
            desc=f"EnzCast [{checkpoint_path.name}]",
            disable=len(ds) < 64,
        ):
            ids = ids.to(device)
            mask = mask.to(device)
            struct = struct.to(device)
            sub = sub.to(device)
            conds = {k: v.to(device) for k, v in conds.items()}

            with amp_ctx:
                out = model(ids, mask, struct, sub, conds)

            for b_idx, row_idx in enumerate(orig_indices):
                row_preds = predictions_by_row.setdefault(int(row_idx), {})
                for task_name in target_tasks:
                    head = TASK_TO_HEAD[task_name]
                    tid = TASK_IDX[task_name]
                    mu_norm = float(out[head][b_idx, 0].float().cpu().item())
                    lv_norm = float(out[head][b_idx, 1].float().cpu().item())
                    mu_log10, var_log10, lv_clip = inverse_gaussian(
                        mu_norm,
                        lv_norm,
                        stats[tid]["mean"],
                        stats[tid]["std"],
                    )
                    row_preds[f"{task_name}_mean_log10"] = float(mu_log10)
                    row_preds[f"{task_name}_var_log10"] = float(var_log10)
                    row_preds[f"{task_name}_std_log10"] = float(np.sqrt(var_log10))
                    row_preds[f"{task_name}_logvar_norm"] = float(lv_clip)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions_by_row


def run_prediction(
    df: pd.DataFrame,
    suite: str = "fulldata",
    tasks: list[str] | None = None,
    model_dir: Path | None = None,
    metadata_path: Path | None = None,
    checkpoint_override: Path | None = None,
    model_key_override: str | None = None,
    esm_config_dir: Path = DEFAULT_ESM_CONFIG_DIR,
    protein_graph_dir: Path | None = None,
    substrate_graph_dir: Path | None = None,
    pdb_dir: Path | None = None,
    device_str: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
    use_autocast: bool = False,
) -> pd.DataFrame:
    tasks = tasks or list(TASK_NAMES)
    suite_cfg = SUITE_DEFAULTS[suite]
    resolved_model_dir = model_dir or (DEFAULT_MODEL_ROOT / suite_cfg["subdir"])
    resolved_meta_path = metadata_path or (resolved_model_dir / suite_cfg["metadata"])
    metadata = load_suite_metadata(resolved_meta_path)

    device = resolve_device(device_str)
    gpu_cfg = detect_gpu_config(0) if device.type == "cuda" else {"batch_size": 4}
    eff_batch_size = batch_size or gpu_cfg["batch_size"]

    merged_preds: dict[int, dict[str, float]] = {}

    if checkpoint_override is not None:
        mkey = model_key_override or ("multitask" if len(tasks) > 1 else tasks[0])
        part = infer_single_checkpoint(
            df=df,
            checkpoint_path=checkpoint_override,
            metadata=metadata,
            model_key=mkey,
            target_tasks=tasks,
            esm_config_dir=esm_config_dir,
            device=device,
            protein_graph_dir=protein_graph_dir,
            substrate_graph_dir=substrate_graph_dir,
            pdb_dir=pdb_dir,
            batch_size=eff_batch_size,
            num_workers=num_workers,
            use_autocast=use_autocast,
        )
        merged_preds = part
    else:
        grouped: dict[tuple[str, str], list[str]] = {}
        for t in tasks:
            route = metadata["task_routing"][t]
            grouped.setdefault((route["checkpoint"], route["model_key"]), []).append(t)

        for (ckpt_name, mkey), ckpt_tasks in grouped.items():
            ckpt_path = resolved_model_dir / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            part = infer_single_checkpoint(
                df=df,
                checkpoint_path=ckpt_path,
                metadata=metadata,
                model_key=mkey,
                target_tasks=ckpt_tasks,
                esm_config_dir=esm_config_dir,
                device=device,
                protein_graph_dir=protein_graph_dir,
                substrate_graph_dir=substrate_graph_dir,
                pdb_dir=pdb_dir,
                batch_size=eff_batch_size,
                num_workers=num_workers,
                use_autocast=use_autocast,
            )
            for r_idx, pdict in part.items():
                merged_preds.setdefault(r_idx, {}).update(pdict)

    out_df = df.copy().reset_index(drop=True)
    pred_cols = []
    for t in tasks:
        pred_cols.extend(
            [
                f"{t}_mean_log10",
                f"{t}_std_log10",
                f"{t}_var_log10",
                f"{t}_logvar_norm",
            ]
        )
    for col in pred_cols:
        out_df[col] = np.nan

    for r_idx, pdict in merged_preds.items():
        for k, v in pdict.items():
            out_df.at[r_idx, k] = v

    return out_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EnzCast kinetic parameter inference")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--suite", choices=["fulldata", "benchmark"], default="fulldata")
    p.add_argument("--tasks", default="all", help="Comma-separated tasks (km,kcat,kcat_km,ki) or 'all'")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-key", choices=["multitask", "km", "kcat", "kcat_km", "ki"], default=None)
    p.add_argument("--esm-config-dir", type=Path, default=DEFAULT_ESM_CONFIG_DIR)
    p.add_argument("--protein-graph-dir", type=Path, default=None)
    p.add_argument("--substrate-graph-dir", type=Path, default=None)
    p.add_argument("--pdb-dir", type=Path, default=None)
    p.add_argument("--ph", type=float, default=None, help="Optional constant pH override for condition scans")
    p.add_argument("--temperature-c", type=float, default=None, help="Optional constant temperature (C) override")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--autocast", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.input_csv, low_memory=False)

    if args.suite == "fulldata":
        if "pH" not in df.columns:
            df["pH"] = 7.0
        if "Temperature_C" not in df.columns:
            df["Temperature_C"] = 37.0
        if "Organism" not in df.columns:
            df["Organism"] = "Homo sapiens"
        if "EnzymeType" not in df.columns:
            df["EnzymeType"] = "wild-type"
        if "EC_Number" not in df.columns:
            df["EC_Number"] = "UNK"

    if args.ph is not None:
        df["pH"] = float(args.ph)
    if args.temperature_c is not None:
        df["Temperature_C"] = float(args.temperature_c)

    if args.tasks.strip().lower() == "all":
        tasks = list(TASK_NAMES)
    else:
        tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in TASK_IDX:
                raise ValueError(f"Unknown task {t!r}; expected one of {TASK_NAMES}")

    out_df = run_prediction(
        df=df,
        suite=args.suite,
        tasks=tasks,
        model_dir=args.model_dir,
        metadata_path=args.metadata,
        checkpoint_override=args.checkpoint,
        model_key_override=args.model_key,
        esm_config_dir=args.esm_config_dir,
        protein_graph_dir=args.protein_graph_dir,
        substrate_graph_dir=args.substrate_graph_dir,
        pdb_dir=args.pdb_dir,
        device_str=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_autocast=args.autocast,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved EnzCast predictions ({len(out_df)} rows) to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
