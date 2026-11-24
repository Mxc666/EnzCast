from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from .featurize import morgan_fingerprint, pdb_to_protein_graph, smiles_to_graph
from .model import SEQ_MAX_LEN

TASK_NAMES = ["km", "kcat", "kcat_km", "ki"]
TASK_IDX = {"km": 0, "kcat": 1, "kcat_km": 2, "ki": 3}
TASK_TO_HEAD = {"km": "km", "kcat": "kcat", "kcat_km": "eff", "ki": "ki"}


def categorise_enzyme_type(val) -> int:
    s = str(val).lower()
    if "wild" in s or "wt" in s or "native" in s:
        return 0
    if "mutant" in s or "mutated" in s or "variant" in s:
        return 1
    return 2


def inverse_gaussian(mu_norm, logvar_norm, train_mean: float, train_std: float):
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError("train_std must be finite and positive")
    mu_arr = np.asarray(mu_norm, dtype=np.float64)
    lv_arr = np.asarray(logvar_norm, dtype=np.float64)
    clipped = np.clip(lv_arr, -8.0, 8.0)
    mu_log10 = mu_arr * train_std + train_mean
    var_log10 = np.exp(clipped) * (train_std**2)
    return mu_log10, var_log10, clipped


def load_suite_metadata(metadata_path: str | Path) -> dict:
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    meta["stats_int"] = {int(k): {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in meta["stats"].items()}
    return meta


def _normalize_protein_graph(struct_data):
    if hasattr(struct_data, "pos") and struct_data.pos is not None:
        pos = struct_data.pos.float()
        pos = torch.nan_to_num(pos, nan=0.0, posinf=100.0, neginf=-100.0)
        if pos.shape[0] > 0:
            pos = pos - pos.mean(dim=0, keepdim=True)
        struct_data.pos = pos.clamp(min=-100.0, max=100.0)

    if struct_data.x.size(0) > SEQ_MAX_LEN:
        struct_data.x = struct_data.x[:SEQ_MAX_LEN]
        struct_data.pos = struct_data.pos[:SEQ_MAX_LEN]
        mask = (struct_data.edge_index[0] < SEQ_MAX_LEN) & (struct_data.edge_index[1] < SEQ_MAX_LEN)
        struct_data.edge_index = struct_data.edge_index[:, mask]
        if hasattr(struct_data, "edge_type") and struct_data.edge_type is not None:
            struct_data.edge_type = struct_data.edge_type[mask]
    return struct_data


class EnzCastDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        protein_graph_dir: str | Path,
        substrate_graph_dir: str | Path,
        recon3d_embeddings_csv: str | Path | None = None,
        max_samples: int | None = None,
        known_stats: dict | None = None,
        ref_ec_map: dict | None = None,
        ref_ec_matrix: torch.Tensor | None = None,
        ref_org_map: dict | None = None,
        ref_unknown_org_id: int | None = None,
        task_filter: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.graph_dir = str(protein_graph_dir)
        self.sub_graph_dir = str(substrate_graph_dir)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        temp_df = pd.read_csv(data_path, low_memory=False)
        req_cols = ["Model_Input_Sequence", "Log10_Value", "SMILES", "UniProt_ID", "Task"]
        temp_df = temp_df.dropna(subset=req_cols)

        temp_df["Task_ID"] = temp_df["Task"].astype(str).str.lower().map(TASK_IDX)
        temp_df = temp_df.dropna(subset=["Task_ID"])
        temp_df["Task_ID"] = temp_df["Task_ID"].astype(int)
        temp_df["Log10_Value"] = pd.to_numeric(temp_df["Log10_Value"], errors="coerce")
        temp_df = temp_df[np.isfinite(temp_df["Log10_Value"])]

        if task_filter is not None:
            tid_target = TASK_IDX[task_filter.lower()]
            temp_df = temp_df[temp_df["Task_ID"] == tid_target]

        available_structs = set(os.listdir(self.graph_dir))
        available_subs = set(os.listdir(self.sub_graph_dir))

        valid_indices = []
        for idx, row in tqdm(temp_df.iterrows(), total=len(temp_df), desc="Indexing graphs", disable=len(temp_df) < 500):
            uid = str(row["UniProt_ID"])
            if f"{uid}.pt" not in available_structs:
                continue
            smi_hash = hashlib.md5(str(row["SMILES"]).encode("utf-8")).hexdigest()
            if f"{smi_hash}.pt" not in available_subs:
                continue
            valid_indices.append(idx)

        self.data = temp_df.loc[valid_indices].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError(f"No valid records with cached graphs found in {data_path}")

        if max_samples:
            self.data = self.data.sample(n=min(max_samples, len(self.data)), random_state=42).reset_index(drop=True)

        if ref_ec_map is not None:
            self.ec_to_id = ref_ec_map
            self.ec_init_matrix = ref_ec_matrix
            self.num_ecs = len(self.ec_to_id)
            self.recon_dim = 128 if ref_ec_matrix is None else int(ref_ec_matrix.shape[1])
        else:
            raw_recon_map = {}
            self.recon_dim = 128
            if recon3d_embeddings_csv is not None and os.path.exists(recon3d_embeddings_csv):
                rdf = pd.read_csv(recon3d_embeddings_csv)
                feat_cols = [c for c in rdf.columns if c.startswith("ReacFeat_")]
                if feat_cols:
                    self.recon_dim = len(feat_cols)
                if "EC_Number_Recon" in rdf.columns:
                    for _, row in rdf.dropna(subset=["EC_Number_Recon"]).iterrows():
                        raw_recon_map[str(row["EC_Number_Recon"]).strip()] = row[feat_cols].values.astype(np.float32)

            all_ecs = np.append(self.data["EC_Number"].astype(str).str.strip().unique(), "UNK")
            self.ec_to_id = {ec: i for i, ec in enumerate(np.unique(all_ecs))}
            self.num_ecs = len(self.ec_to_id)

            init_mat = np.random.normal(0, 0.02, (self.num_ecs, self.recon_dim)).astype(np.float32)
            for ec, idx in self.ec_to_id.items():
                if ec in raw_recon_map:
                    init_mat[idx] = raw_recon_map[ec]
            self.ec_init_matrix = torch.tensor(init_mat, dtype=torch.float32)

        if ref_org_map is not None:
            self.org_map = ref_org_map
            self.unknown_org_id = len(ref_org_map) if ref_unknown_org_id is None else int(ref_unknown_org_id)
            self.num_organisms = self.unknown_org_id + 1
        else:
            unique_orgs = self.data["Organism"].astype(str).unique()
            self.org_map = {name: i for i, name in enumerate(unique_orgs)}
            self.unknown_org_id = len(unique_orgs)
            self.num_organisms = len(unique_orgs) + 1

        self.data["EnzymeType"] = self.data["EnzymeType"].fillna("unknown").astype(str).str.lower()
        self.data["Type_ID"] = self.data["EnzymeType"].apply(categorise_enzyme_type)
        self.num_types = 3

        self.task_counts = {tid: int((self.data["Task_ID"] == tid).sum()) for tid in [0, 1, 2, 3]}
        if known_stats is not None:
            self.stats = {int(k): v for k, v in known_stats.items()}
        else:
            self.stats = {}
            for tid in [0, 1, 2, 3]:
                vals = self.data[self.data["Task_ID"] == tid]["Log10_Value"]
                if len(vals) > 1:
                    std_val = float(vals.std())
                    self.stats[tid] = {"mean": float(vals.mean()), "std": 1.0 if std_val < 1e-6 else std_val}
                elif len(vals) == 1:
                    self.stats[tid] = {"mean": float(vals.iloc[0]), "std": 1.0}
                else:
                    self.stats[tid] = {"mean": 0.0, "std": 1.0}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        seq = str(row["Model_Input_Sequence"])[:SEQ_MAX_LEN]
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=SEQ_MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        uid = str(row["UniProt_ID"])
        struct_data = torch.load(os.path.join(self.graph_dir, f"{uid}.pt"), map_location="cpu", weights_only=False)
        struct_data = _normalize_protein_graph(struct_data)

        smi_str = str(row["SMILES"])
        smi_hash = hashlib.md5(smi_str.encode("utf-8")).hexdigest()
        sub_graph = torch.load(os.path.join(self.sub_graph_dir, f"{smi_hash}.pt"), map_location="cpu", weights_only=False)
        if not torch.isfinite(sub_graph.x).all():
            sub_graph.x = torch.nan_to_num(sub_graph.x, nan=0.0)
        fp = morgan_fingerprint(smi_str)

        ec = str(row["EC_Number"]).strip()
        ec_id = self.ec_to_id.get(ec, self.ec_to_id.get("UNK", 0))

        ph = float(row["pH"]) if pd.notna(row["pH"]) else 7.0
        ph_mask = 1.0 if pd.notna(row["pH"]) else 0.0
        temp = float(row["Temperature_C"]) if pd.notna(row["Temperature_C"]) else 25.0
        temp_mask = 1.0 if pd.notna(row["Temperature_C"]) else 0.0

        org_id = self.org_map.get(str(row["Organism"]), self.unknown_org_id)
        type_id = int(row["Type_ID"])

        cond = {
            "ph": torch.tensor(ph, dtype=torch.float32),
            "ph_mask": torch.tensor(ph_mask, dtype=torch.float32),
            "temp": torch.tensor(temp, dtype=torch.float32),
            "temp_mask": torch.tensor(temp_mask, dtype=torch.float32),
            "species": torch.tensor(org_id, dtype=torch.long),
            "type": torch.tensor(type_id, dtype=torch.long),
            "ec_id": torch.tensor(ec_id, dtype=torch.long),
            "fp": fp,
        }

        val_log = float(row["Log10_Value"])
        task_id = int(row["Task_ID"])
        mean, std = self.stats[task_id]["mean"], self.stats[task_id]["std"]
        val_norm = (val_log - mean) / std
        return (
            input_ids,
            attn_mask,
            struct_data,
            sub_graph,
            cond,
            torch.tensor([val_norm], dtype=torch.float32),
            torch.tensor(task_id, dtype=torch.long),
        )


class EnzCastInferenceDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        metadata: dict,
        model_key: str,
        protein_graph_dir: str | Path | None = None,
        substrate_graph_dir: str | Path | None = None,
        pdb_dir: str | Path | None = None,
        default_missing_temp_c: float = 25.0,
        mask_mode: str = "observed",
    ):
        self.tokenizer = tokenizer
        self.protein_graph_dir = Path(protein_graph_dir) if protein_graph_dir else None
        self.substrate_graph_dir = Path(substrate_graph_dir) if substrate_graph_dir else None
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.ec_to_id = metadata["ec_to_id"]
        org_spec = metadata["organism_maps"][model_key]
        self.org_map = org_spec["map"]
        self.unknown_org_id = int(org_spec["unknown_id"])
        self.default_missing_temp_c = float(default_missing_temp_c)
        self.mask_mode = mask_mode

        df = dataframe.reset_index(drop=True).copy()
        for req in ["Model_Input_Sequence", "SMILES", "UniProt_ID"]:
            if req not in df.columns:
                raise ValueError(f"Input table missing required column: {req!r}")

        valid_rows = []
        self._protein_graphs = {}
        self._substrate_graphs = {}

        for idx, row in df.iterrows():
            seq = str(row["Model_Input_Sequence"]).strip() if pd.notna(row["Model_Input_Sequence"]) else ""
            smi = str(row["SMILES"]).strip() if pd.notna(row["SMILES"]) else ""
            uid = str(row["UniProt_ID"]).strip() if pd.notna(row["UniProt_ID"]) else ""
            if not seq or not smi or not uid:
                continue

            prot_pt = (
                self.protein_graph_dir / f"{uid}.pt"
                if self.protein_graph_dir is not None
                else None
            )
            if prot_pt is not None and prot_pt.exists():
                pass
            elif uid in self._protein_graphs:
                pass
            else:
                pdb_candidate = None
                if "PDB_Path" in row and pd.notna(row["PDB_Path"]):
                    pdb_candidate = Path(str(row["PDB_Path"]))
                elif self.pdb_dir is not None and (self.pdb_dir / f"{uid}.pdb").exists():
                    pdb_candidate = self.pdb_dir / f"{uid}.pdb"
                if pdb_candidate is None or not pdb_candidate.exists():
                    continue
                built = pdb_to_protein_graph(pdb_candidate)
                if built is None:
                    continue
                self._protein_graphs[uid] = built

            smi_hash = hashlib.md5(smi.encode("utf-8")).hexdigest()
            sub_pt = (
                self.substrate_graph_dir / f"{smi_hash}.pt"
                if self.substrate_graph_dir is not None
                else None
            )
            if sub_pt is not None and sub_pt.exists():
                pass
            elif smi_hash in self._substrate_graphs:
                pass
            else:
                built_sub = smiles_to_graph(smi)
                if built_sub is None:
                    continue
                self._substrate_graphs[smi_hash] = built_sub

            valid_rows.append(idx)

        if not valid_rows:
            raise ValueError("No valid inference records with resolvable protein and substrate graphs")
        self.valid_input_indices = valid_rows
        self.data = df.loc[valid_rows].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        seq = str(row["Model_Input_Sequence"])[:SEQ_MAX_LEN]
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=SEQ_MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        uid = str(row["UniProt_ID"]).strip()
        if self.protein_graph_dir is not None and (self.protein_graph_dir / f"{uid}.pt").exists():
            struct_data = torch.load(
                self.protein_graph_dir / f"{uid}.pt",
                map_location="cpu",
                weights_only=False,
            )
        else:
            struct_data = self._protein_graphs[uid].clone()
        struct_data = _normalize_protein_graph(struct_data)

        smi = str(row["SMILES"]).strip()
        smi_hash = hashlib.md5(smi.encode("utf-8")).hexdigest()
        if self.substrate_graph_dir is not None and (self.substrate_graph_dir / f"{smi_hash}.pt").exists():
            sub_graph = torch.load(
                self.substrate_graph_dir / f"{smi_hash}.pt",
                map_location="cpu",
                weights_only=False,
            )
        else:
            sub_graph = self._substrate_graphs[smi_hash].clone()
        if not torch.isfinite(sub_graph.x).all():
            sub_graph.x = torch.nan_to_num(sub_graph.x, nan=0.0)
        fp = morgan_fingerprint(smi)

        ec = str(row.get("EC_Number", "UNK")).strip() if pd.notna(row.get("EC_Number", "UNK")) else "UNK"
        ec_id = self.ec_to_id.get(ec, self.ec_to_id.get("UNK", 0))

        has_ph = "pH" in row and pd.notna(row["pH"])
        ph = float(row["pH"]) if has_ph else 7.0
        ph_mask = 1.0 if has_ph else 0.0

        has_temp = "Temperature_C" in row and pd.notna(row["Temperature_C"])
        temp = float(row["Temperature_C"]) if has_temp else self.default_missing_temp_c
        temp_mask = 1.0 if has_temp else 0.0

        if self.mask_mode in ("mask_ph", "mask_both"):
            ph, ph_mask = 7.0, 0.0
        if self.mask_mode in ("mask_temp", "mask_both"):
            temp, temp_mask = 25.0, 0.0
        if self.mask_mode == "fixed_temp25":
            temp, temp_mask = 25.0, 1.0

        org_str = str(row.get("Organism", "Homo sapiens")) if pd.notna(row.get("Organism", "Homo sapiens")) else "unknown"
        org_id = self.org_map.get(org_str, self.unknown_org_id)

        enz_type = row.get("EnzymeType", "wild-type")
        type_id = categorise_enzyme_type(enz_type if pd.notna(enz_type) else "unknown")

        cond = {
            "ph": torch.tensor(ph, dtype=torch.float32),
            "ph_mask": torch.tensor(ph_mask, dtype=torch.float32),
            "temp": torch.tensor(temp, dtype=torch.float32),
            "temp_mask": torch.tensor(temp_mask, dtype=torch.float32),
            "species": torch.tensor(org_id, dtype=torch.long),
            "type": torch.tensor(type_id, dtype=torch.long),
            "ec_id": torch.tensor(ec_id, dtype=torch.long),
            "fp": fp,
        }
        return input_ids, attn_mask, struct_data, sub_graph, cond, self.valid_input_indices[idx]


def collate_fn(batch):
    ids, masks, structs, subs, conds_list, vals, tasks = zip(*batch)
    conds = {
        "ph": torch.stack([c["ph"] for c in conds_list]),
        "ph_mask": torch.stack([c["ph_mask"] for c in conds_list]).unsqueeze(-1),
        "temp": torch.stack([c["temp"] for c in conds_list]),
        "temp_mask": torch.stack([c["temp_mask"] for c in conds_list]).unsqueeze(-1),
        "species": torch.stack([c["species"] for c in conds_list]),
        "type": torch.stack([c["type"] for c in conds_list]),
        "ec_id": torch.stack([c["ec_id"] for c in conds_list]),
        "fp": torch.stack([c["fp"] for c in conds_list]),
    }
    return (
        torch.stack(ids),
        torch.stack(masks),
        Batch.from_data_list(structs),
        Batch.from_data_list(subs),
        conds,
        torch.stack(vals),
        torch.stack(tasks),
    )


def collate_fn_inference(batch):
    ids, masks, structs, subs, conds_list, indices = zip(*batch)
    conds = {
        "ph": torch.stack([c["ph"] for c in conds_list]),
        "ph_mask": torch.stack([c["ph_mask"] for c in conds_list]).unsqueeze(-1),
        "temp": torch.stack([c["temp"] for c in conds_list]),
        "temp_mask": torch.stack([c["temp_mask"] for c in conds_list]).unsqueeze(-1),
        "species": torch.stack([c["species"] for c in conds_list]),
        "type": torch.stack([c["type"] for c in conds_list]),
        "ec_id": torch.stack([c["ec_id"] for c in conds_list]),
        "fp": torch.stack([c["fp"] for c in conds_list]),
    }
    return (
        torch.stack(ids),
        torch.stack(masks),
        Batch.from_data_list(structs),
        Batch.from_data_list(subs),
        conds,
        list(indices),
    )
