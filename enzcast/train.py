#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from enzcast.dataset import TASK_IDX, TASK_NAMES, TASK_TO_HEAD, EnzCastDataset, collate_fn
from enzcast.model import (
    BETA_NLL_BETA,
    COND_DIM,
    DROPOUT,
    EMA_DECAY,
    ESM_EXTRACT_LAYERS,
    HIDDEN_DIM,
    SEQ_MAX_LEN,
    SUBSTRATE_IN_DIM,
    BetaNLLLoss,
    EMAModel,
    EnzCastModel,
    UncertaintyWeightedMTL,
    consistency_loss_log10,
    detect_gpu_config,
    get_nll_weight_short,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_config_only_esm_dir(esm_model: str) -> bool:
    p = Path(esm_model)
    if not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return has_cfg and not has_weights


def evaluate_epoch(model, loader, stats, device, target_tasks: list[str], ema: EMAModel | None = None) -> dict[str, dict[str, float]]:
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    records = []
    with torch.no_grad():
        for ids, mask, struct, sub, conds, truth, tasks in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            struct = struct.to(device)
            sub = sub.to(device)
            conds = {k: v.to(device) for k, v in conds.items()}

            out = model(ids, mask, struct, sub, conds)
            preds_all = torch.stack(
                [out["km"][:, 0], out["kcat"][:, 0], out["eff"][:, 0], out["ki"][:, 0]],
                dim=1,
            ).cpu().float().numpy()
            truth_np = truth.cpu().float().numpy().ravel()
            tasks_np = tasks.cpu().numpy().ravel()

            for i, tid in enumerate(tasks_np):
                tid_int = int(tid)
                tname = TASK_NAMES[tid_int]
                if tname not in target_tasks:
                    continue
                mean, std = stats[tid_int]["mean"], stats[tid_int]["std"]
                records.append(
                    {
                        "task": tname,
                        "pred": float(preds_all[i, tid_int]) * std + mean,
                        "true": float(truth_np[i]) * std + mean,
                    }
                )

    if ema is not None:
        ema.restore(model)

    df = pd.DataFrame(records)
    metrics = {}
    if not df.empty:
        for tname in target_tasks:
            sub_df = df[df["task"] == tname]
            if len(sub_df) > 1:
                metrics[tname] = {
                    "n": int(len(sub_df)),
                    "r2": float(r2_score(sub_df["true"], sub_df["pred"])),
                    "rmse": float(np.sqrt(mean_squared_error(sub_df["true"], sub_df["pred"]))),
                }
    return metrics


def run_contrastive_pretraining(
    model,
    loader,
    device,
    epochs: int,
    gpu_cfg: dict,
    grad_accum_steps: int,
    save_path: Path,
    max_steps: int | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_scaler = bool(gpu_cfg["use_scaler"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_cfg["precision"] == "bfloat16" else torch.float16

    model.train()
    num_batches = len(loader)
    step_count = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (ids, mask, struct, sub, _, _, _) in enumerate(tqdm(loader, desc=f"Pretrain {epoch + 1}/{epochs}")):
            ids = ids.to(device)
            mask = mask.to(device)
            struct = struct.to(device)
            sub = sub.to(device)
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits = model.contrastive_forward(ids, mask, struct, sub, autocast_dtype)
                if not torch.isfinite(logits).all():
                    continue
                labels = torch.arange(logits.size(0), device=device)
                loss = ((loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2.0) / grad_accum_steps

            if not torch.isfinite(loss):
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    break
        if max_steps is not None and step_count >= max_steps:
            break

    torch.save(model.state_dict(), save_path)


def train_stage1(model, train_dl, val_dl, stats, device, args, gpu_cfg: dict, target_tasks: list[str], out_dir: Path, prefix: str):
    use_scaler = bool(gpu_cfg["use_scaler"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_cfg["precision"] == "bfloat16" else torch.float16

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prediction_head" in name or "fusion" in name or "cond_enc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )

    est_steps = max(1, len(train_dl) * max(1, args.stage1_epochs) // max(1, args.grad_accum_steps))
    total_steps = min(est_steps, args.max_steps) if args.max_steps is not None else est_steps
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    loss_fn = BetaNLLLoss(beta=BETA_NLL_BETA)
    mtl_loss = UncertaintyWeightedMTL(n_tasks=4).to(device) if args.mode == "multitask" else None
    opt_mtl = torch.optim.AdamW(mtl_loss.parameters(), lr=1e-3) if mtl_loss is not None else None
    ema = EMAModel(model, decay=EMA_DECAY)

    best_ema_score = -float("inf")
    best_reg_score = -float("inf")
    epochs_no_improve = 0
    opt_steps = 0
    ema_path = out_dir / f"{prefix}.pt"
    reg_path = out_dir / f"{prefix}_regular.pt"
    num_batches = len(train_dl)

    for epoch in range(args.stage1_epochs):
        model.train()
        nll_w = get_nll_weight_short(epoch, args.stage1_epochs)
        opt.zero_grad(set_to_none=True)
        if opt_mtl is not None:
            opt_mtl.zero_grad(set_to_none=True)
        accum_count = 0

        for batch_idx, (ids, mask, struct, sub, conds, truth_val, task_ids) in enumerate(
            tqdm(train_dl, desc=f"Stage1 {epoch + 1}/{args.stage1_epochs}")
        ):
            ids = ids.to(device)
            mask = mask.to(device)
            struct = struct.to(device)
            sub = sub.to(device)
            truth_val = truth_val.to(device)
            task_ids = task_ids.to(device)
            conds = {k: v.to(device) for k, v in conds.items()}

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                out = model(ids, mask, struct, sub, conds, autocast_dtype=autocast_dtype)
                if args.mode == "multitask":
                    losses_dict = {}
                    for tname in TASK_NAMES:
                        tid = TASK_IDX[tname]
                        head = TASK_TO_HEAD[tname]
                        sel = task_ids == tid
                        if sel.any():
                            losses_dict[head] = loss_fn(out[head][sel, 0], out[head][sel, 1], truth_val[sel], nll_w)
                    if not losses_dict:
                        continue
                    model_loss = mtl_loss(losses_dict)
                    if (task_ids == 2).any() or ((task_ids == 1).any() and (task_ids == 0).any()):
                        cons = consistency_loss_log10(out["kcat"][:, 0], out["km"][:, 0], out["eff"][:, 0], stats)
                        model_loss = model_loss + args.consistency_weight * cons
                else:
                    tname = target_tasks[0]
                    head = TASK_TO_HEAD[tname]
                    model_loss = loss_fn(out[head][:, 0], out[head][:, 1], truth_val, nll_w)

                model_loss = model_loss / args.grad_accum_steps

            if not torch.isfinite(model_loss):
                continue

            if use_scaler:
                scaler.scale(model_loss).backward()
            else:
                model_loss.backward()
            accum_count += 1

            is_last_batch = (batch_idx + 1) == num_batches
            reach_limit = args.max_steps is not None and (opt_steps + 1) >= args.max_steps and accum_count > 0
            if (accum_count % args.grad_accum_steps == 0) or is_last_batch or reach_limit:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if opt_mtl is not None:
                    opt_mtl.step()
                scheduler.step()
                ema.update(model)
                opt.zero_grad(set_to_none=True)
                if opt_mtl is not None:
                    opt_mtl.zero_grad(set_to_none=True)
                opt_steps += 1
                if args.max_steps is not None and opt_steps >= args.max_steps:
                    break

        if val_dl is not None:
            ema_metrics = evaluate_epoch(model, val_dl, stats, device, target_tasks, ema=ema)
            ema_score = float(np.mean([m["r2"] for m in ema_metrics.values()])) if ema_metrics else -1e9
            if ema_score > best_ema_score:
                best_ema_score = ema_score
                ema.apply_shadow(model)
                torch.save(model.state_dict(), ema_path)
                ema.restore(model)

            reg_metrics = evaluate_epoch(model, val_dl, stats, device, target_tasks, ema=None)
            reg_score = float(np.mean([m["r2"] for m in reg_metrics.values()])) if reg_metrics else -1e9
            if reg_score > best_reg_score:
                best_reg_score = reg_score
                epochs_no_improve = 0
                torch.save(model.state_dict(), reg_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    break
        else:
            ema.apply_shadow(model)
            torch.save(model.state_dict(), ema_path)
            ema.restore(model)
            torch.save(model.state_dict(), reg_path)

        if args.max_steps is not None and opt_steps >= args.max_steps:
            break

    if not ema_path.exists():
        ema.apply_shadow(model)
        torch.save(model.state_dict(), ema_path)
        ema.restore(model)
    if not reg_path.exists():
        torch.save(model.state_dict(), reg_path)

    return ema_path, opt_steps


def build_parser() -> argparse.ArgumentParser:
    default_dataset_root = CODE_ROOT.parent / "Dataset" / "kinbench"
    p = argparse.ArgumentParser(description="Train EnzCast multi-task or single-task kinetic models")
    p.add_argument("--mode", choices=["multitask", "singletask"], default="multitask")
    p.add_argument("--task", choices=TASK_NAMES, default="km")
    p.add_argument("--data-mode", choices=["benchmark", "fulldata"], default="benchmark")
    p.add_argument("--split-dir", type=Path, default=default_dataset_root)
    p.add_argument("--train-csv", type=Path, default=None, help="Optional explicit training CSV overriding --split-dir")
    p.add_argument("--val-csv", type=Path, default=None, help="Optional explicit validation CSV overriding --split-dir")
    p.add_argument("--protein-graph-dir", type=Path, required=True)
    p.add_argument("--substrate-graph-dir", type=Path, required=True)
    p.add_argument("--recon3d-embeddings", type=Path, default=default_dataset_root / "recon3d_reaction_embeddings.csv")
    p.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="Optional EnzCast checkpoint for warm-start or fine-tuning")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="Optional maximum optimizer steps per training stage")
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    gpu_cfg = detect_gpu_config(0) if device.type == "cuda" else detect_gpu_config()
    batch_size = args.batch_size or gpu_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else gpu_cfg["num_workers"]

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)

    if args.train_csv is not None:
        primary_train_csv = args.train_csv
        val_csv = args.val_csv
    else:
        train_csv = args.split_dir / "kinbench_train.csv"
        val_csv = args.val_csv or (args.split_dir / "kinbench_val.csv")
        test_csv = args.split_dir / "kinbench_test.csv"
        if args.data_mode == "fulldata":
            merged_path = args.output_dir / "_merged_fulldata_train.csv"
            merged_df = pd.concat(
                [pd.read_csv(train_csv, low_memory=False), pd.read_csv(test_csv, low_memory=False)],
                ignore_index=True,
            )
            merged_df.to_csv(merged_path, index=False)
            primary_train_csv = merged_path
        else:
            primary_train_csv = train_csv

    full_train_ds = EnzCastDataset(
        data_path=primary_train_csv,
        tokenizer=tokenizer,
        protein_graph_dir=args.protein_graph_dir,
        substrate_graph_dir=args.substrate_graph_dir,
        recon3d_embeddings_csv=args.recon3d_embeddings if args.recon3d_embeddings.exists() else None,
        max_samples=args.max_samples,
        task_filter=None,
    )

    if args.mode == "singletask":
        target_tasks = [args.task]
        train_ds = EnzCastDataset(
            data_path=primary_train_csv,
            tokenizer=tokenizer,
            protein_graph_dir=args.protein_graph_dir,
            substrate_graph_dir=args.substrate_graph_dir,
            known_stats=full_train_ds.stats,
            ref_ec_map=full_train_ds.ec_to_id,
            ref_ec_matrix=full_train_ds.ec_init_matrix,
            max_samples=args.max_samples,
            task_filter=args.task,
        )
        val_ds = (
            EnzCastDataset(
                data_path=val_csv,
                tokenizer=tokenizer,
                protein_graph_dir=args.protein_graph_dir,
                substrate_graph_dir=args.substrate_graph_dir,
                known_stats=full_train_ds.stats,
                ref_ec_map=full_train_ds.ec_to_id,
                ref_ec_matrix=full_train_ds.ec_init_matrix,
                ref_org_map=train_ds.org_map,
                ref_unknown_org_id=train_ds.unknown_org_id,
                max_samples=args.max_samples,
                task_filter=args.task,
            )
            if (val_csv is not None and val_csv.exists())
            else None
        )
        prefix = f"enzcast_{'fulldata_' if args.data_mode == 'fulldata' else ''}{args.task}"
        model_key = args.task
    else:
        target_tasks = list(TASK_NAMES)
        train_ds = full_train_ds
        val_ds = (
            EnzCastDataset(
                data_path=val_csv,
                tokenizer=tokenizer,
                protein_graph_dir=args.protein_graph_dir,
                substrate_graph_dir=args.substrate_graph_dir,
                known_stats=full_train_ds.stats,
                ref_ec_map=full_train_ds.ec_to_id,
                ref_ec_matrix=full_train_ds.ec_init_matrix,
                ref_org_map=full_train_ds.org_map,
                ref_unknown_org_id=full_train_ds.unknown_org_id,
                max_samples=args.max_samples,
                task_filter=None,
            )
            if (val_csv is not None and val_csv.exists())
            else None
        )
        prefix = f"enzcast_{'fulldata_' if args.data_mode == 'fulldata' else ''}multitask"
        model_key = "multitask"

    from_cfg_only = _is_config_only_esm_dir(args.esm_model)
    model = EnzCastModel(
        args.esm_model,
        num_species=full_train_ds.num_organisms,
        num_types=full_train_ds.num_types,
        num_ecs=full_train_ds.num_ecs,
        recon_dim=full_train_ds.recon_dim,
        ec_init_matrix=full_train_ds.ec_init_matrix,
        from_config_only=from_cfg_only,
    ).to(device)

    if args.init_checkpoint is not None:
        init_state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        current_state = model.state_dict()
        compatible = {k: v for k, v in init_state.items() if k in current_state and current_state[k].shape == v.shape}
        current_state.update(compatible)
        model.load_state_dict(current_state, strict=True)
        del init_state

    pretrain_dl = DataLoader(
        full_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=len(full_train_ds) > batch_size,
    )
    if args.pretrain_epochs > 0:
        run_contrastive_pretraining(
            model,
            pretrain_dl,
            device,
            args.pretrain_epochs,
            gpu_cfg,
            args.grad_accum_steps,
            args.output_dir / f"{prefix}_pretrained.pt",
            max_steps=args.max_steps,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=len(train_ds) > batch_size,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        if val_ds is not None
        else None
    )

    ema_ckpt, opt_steps = train_stage1(
        model,
        train_dl,
        val_dl,
        full_train_ds.stats,
        device,
        args,
        gpu_cfg,
        target_tasks,
        args.output_dir,
        prefix,
    )

    task_routing = {}
    for tname in target_tasks:
        task_routing[tname] = {
            "checkpoint": ema_ckpt.name,
            "head": TASK_TO_HEAD[tname],
            "model_key": model_key,
            "task_id": TASK_IDX[tname],
        }

    metadata_out = {
        "suite": args.data_mode,
        "optimizer_steps_completed": int(opt_steps),
        "model_architecture": {
            "seq_max_len": SEQ_MAX_LEN,
            "substrate_in_dim": SUBSTRATE_IN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "cond_dim": COND_DIM,
            "dropout": DROPOUT,
            "esm_extract_layers": ESM_EXTRACT_LAYERS,
            "num_embedding_organisms": full_train_ds.num_organisms,
            "num_types": full_train_ds.num_types,
            "num_ecs": full_train_ds.num_ecs,
            "recon_dim": full_train_ds.recon_dim,
        },
        "task_routing": task_routing,
        "stats": {str(k): v for k, v in full_train_ds.stats.items()},
        "train_counts": {str(k): v for k, v in full_train_ds.task_counts.items()},
        "ec_to_id": full_train_ds.ec_to_id,
        "organism_maps": {
            "multitask": {"unknown_id": full_train_ds.unknown_org_id, "map": full_train_ds.org_map},
            model_key: {"unknown_id": train_ds.unknown_org_id, "map": train_ds.org_map},
        },
    }
    meta_file = args.output_dir / f"{prefix}_metadata.json"
    meta_file.write_text(json.dumps(metadata_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Training finished ({opt_steps} optimizer steps). Saved EMA checkpoint: {ema_ckpt} and metadata: {meta_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
