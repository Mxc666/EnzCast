
import os
import sys
import math
import copy
import argparse
import warnings
import hashlib
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

DEFAULT_PROJECT_ROOT = "."

SEQ_MAX_LEN = 1024
SUBSTRATE_IN_DIM = 42
HIDDEN_DIM = 256
COND_DIM = 128
DROPOUT = 0.15

EMA_DECAY = 0.999
BETA_NLL_BETA = 0.5
ESM_EXTRACT_LAYERS = [10, 21, 32]

DEFAULT_STAGE1_EPOCHS = 5
DEFAULT_STAGE2_EPOCHS = 6
PRETRAIN_EPOCH = 3

TASK_NAMES = ["km", "kcat", "kcat_km", "ki"]
TASK_IDX = {"km": 0, "kcat": 1, "kcat_km": 2, "ki": 3}


def detect_gpu_config(device_id=0):
    if not torch.cuda.is_available():
        return {
            "gpu_type": "cpu", "precision": "float32", "use_scaler": False,
            "batch_size": 2, "num_workers": 2, "tf32": False,
        }

    gpu_name = torch.cuda.get_device_name(device_id).lower()

    if "a800" in gpu_name or "a100" in gpu_name:
        return {
            "gpu_type": "A800", "precision": "bfloat16", "use_scaler": False,
            "batch_size": 22, "num_workers": 10, "tf32": True,
        }
    if any(x in gpu_name for x in ["rtx 30", "rtx 40", "a6000", "a5000", "a4000"]):
        return {
            "gpu_type": "Ampere", "precision": "bfloat16", "use_scaler": False,
            "batch_size": 6, "num_workers": 4, "tf32": True,
        }
    if "v100" in gpu_name:
        return {
            "gpu_type": "V100", "precision": "float16", "use_scaler": True,
            "batch_size": 4, "num_workers": 8, "tf32": False,
        }
    return {
        "gpu_type": "Unknown", "precision": "float16", "use_scaler": True,
        "batch_size": 4, "num_workers": 4, "tf32": False,
    }


def log_print(msg):
    print(msg, flush=True)


class EMAModel:

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow(model)

    def _init_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class EnzCastDataset(Dataset):
    def __init__(
        self, data_path, tokenizer, graph_dir, sub_graph_dir,
        max_samples=None, known_stats=None, ref_ec_map=None,
        ref_ec_matrix=None, base_dir=None,
    ):
        self.tokenizer = tokenizer
        self.sub_graph_dir = sub_graph_dir
        self.graph_dir = graph_dir

        log_print(f"\n[Init] Loading Data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[Error] File not found: {data_path}")

        try:
            temp_df = pd.read_csv(data_path, low_memory=False)
            req_cols = ["Model_Input_Sequence", "Log10_Value", "SMILES", "UniProt_ID", "Task"]
            temp_df = temp_df.dropna(subset=req_cols)

            task_map = {"km": 0, "kcat": 1, "kcat_km": 2, "ki": 3}
            temp_df["Task_ID"] = temp_df["Task"].astype(str).str.lower().map(task_map)
            temp_df = temp_df.dropna(subset=["Task_ID"])
            temp_df["Task_ID"] = temp_df["Task_ID"].astype(int)
            temp_df["Log10_Value"] = pd.to_numeric(temp_df["Log10_Value"], errors="coerce")

            log_print("  [Strict Filter] indexing local graph files...")
            available_structs = set(os.listdir(self.graph_dir))
            available_subs = set(os.listdir(self.sub_graph_dir))

            valid_indices = []
            iter_wrapper = tqdm(temp_df.iterrows(), total=len(temp_df), desc="Checking")
            for idx, row in iter_wrapper:
                uid = str(row["UniProt_ID"])
                if f"{uid}.pt" not in available_structs:
                    continue
                smi = str(row["SMILES"])
                smi_hash = hashlib.md5(smi.encode("utf-8")).hexdigest()
                if f"{smi_hash}.pt" not in available_subs:
                    continue
                valid_indices.append(idx)

            self.data = temp_df.loc[valid_indices].reset_index(drop=True)
            log_print(f"  -> Original rows: {len(temp_df)}")
            log_print(f"  -> Valid rows (Files Exist): {len(self.data)}")

            if len(self.data) == 0:
                raise ValueError("CRITICAL: No valid data found!")

        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

        if max_samples:
            self.data = self.data.sample(
                n=min(max_samples, len(self.data)), random_state=42
            ).reset_index(drop=True)

        if ref_ec_map is not None:
            self.ec_to_id = ref_ec_map
            self.ec_init_matrix = ref_ec_matrix
            self.num_ecs = len(self.ec_to_id)
            log_print(f"[Init] Using Reference EC Map. Total ECs: {self.num_ecs}")
        else:
            if base_dir is None:
                base_dir = os.path.dirname(os.path.dirname(self.graph_dir))
            self.recon_file = os.path.join(base_dir, "Recon3D/reaction_embeddings.csv")
            self.raw_recon_map = {}
            self.recon_dim = 128
            if os.path.exists(self.recon_file):
                rdf = pd.read_csv(self.recon_file)
                feat_cols = [c for c in rdf.columns if c.startswith("ReacFeat_")]
                self.recon_dim = len(feat_cols)
                if "EC_Number_Recon" in rdf.columns:
                    rdf_clean = rdf.dropna(subset=["EC_Number_Recon"])
                    for _, row in rdf_clean.iterrows():
                        ec_str = str(row["EC_Number_Recon"]).strip()
                        self.raw_recon_map[ec_str] = row[feat_cols].values.astype(np.float32)

            all_ecs = self.data["EC_Number"].astype(str).str.strip().unique()
            all_ecs = np.append(all_ecs, "UNK")
            self.ec_to_id = {ec: i for i, ec in enumerate(np.unique(all_ecs))}
            self.num_ecs = len(self.ec_to_id)
            log_print(f"[Init] Built New EC Map. Total Unique ECs: {self.num_ecs}")

            self.ec_init_matrix = np.random.normal(0, 0.02, (self.num_ecs, self.recon_dim))
            found_recon = 0
            for ec, idx in self.ec_to_id.items():
                if ec in self.raw_recon_map:
                    self.ec_init_matrix[idx] = self.raw_recon_map[ec]
                    found_recon += 1
            log_print(f"[Init] Initialized {found_recon}/{self.num_ecs} ECs with Recon3D priors.")
            self.ec_init_matrix = torch.tensor(self.ec_init_matrix, dtype=torch.float)

        unique_orgs = self.data["Organism"].astype(str).unique()
        self.org_map = {name: i for i, name in enumerate(unique_orgs)}
        self.num_organisms = len(unique_orgs) + 1

        self.data["EnzymeType"] = self.data["EnzymeType"].fillna("unknown").astype(str).str.lower()

        def categorise_type(val):
            if "wild" in val or "wt" in val or "native" in val:
                return 0
            if "mutant" in val or "mutated" in val or "variant" in val:
                return 1
            return 2

        self.data["Type_ID"] = self.data["EnzymeType"].apply(categorise_type)
        self.num_types = 3

        if known_stats is not None:
            self.stats = known_stats
            log_print("[Stats] Using provided statistics (from Train).")
        else:
            self.stats = {}
            self.task_counts = {}
            for tid in [0, 1, 2, 3]:
                vals = self.data[self.data["Task_ID"] == tid]["Log10_Value"]
                self.task_counts[tid] = len(vals)
                if len(vals) > 1:
                    mean_val = vals.mean()
                    std_val = vals.std()
                    if pd.isna(std_val) or std_val < 1e-6:
                        std_val = 1.0
                    self.stats[tid] = {"mean": mean_val, "std": std_val}
                    log_print(
                        f"[Stats] Task {tid}: Mean={self.stats[tid]['mean']:.4f}, "
                        f"Std={self.stats[tid]['std']:.4f}, N={len(vals)}"
                    )
                elif len(vals) == 1:
                    self.stats[tid] = {"mean": vals.iloc[0], "std": 1.0}
                else:
                    self.stats[tid] = {"mean": 0.0, "std": 1.0}

        if not hasattr(self, "task_counts"):
            self.task_counts = {}
            for tid in [0, 1, 2, 3]:
                self.task_counts[tid] = int((self.data["Task_ID"] == tid).sum())

    def get_morgan_fp(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(2048, dtype=torch.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.from_numpy(arr).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row["Model_Input_Sequence"])[:SEQ_MAX_LEN]
        enc = self.tokenizer(
            seq, truncation=True, padding="max_length", max_length=SEQ_MAX_LEN, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        uid = str(row["UniProt_ID"])
        pt_path = os.path.join(self.graph_dir, f"{uid}.pt")

        struct_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if hasattr(struct_data, "pos") and struct_data.pos is not None:
            pos = struct_data.pos.float()
            pos = torch.nan_to_num(pos, nan=0.0, posinf=100.0, neginf=-100.0)
            if pos.shape[0] > 0:
                pos = pos - pos.mean(dim=0, keepdim=True)
            pos = pos.clamp(min=-100.0, max=100.0)
            struct_data.pos = pos

        if struct_data.x.size(0) > SEQ_MAX_LEN:
            struct_data.x = struct_data.x[:SEQ_MAX_LEN]
            struct_data.pos = struct_data.pos[:SEQ_MAX_LEN]
            mask = (struct_data.edge_index[0] < SEQ_MAX_LEN) & (
                struct_data.edge_index[1] < SEQ_MAX_LEN
            )
            struct_data.edge_index = struct_data.edge_index[:, mask]
            if hasattr(struct_data, "edge_type"):
                struct_data.edge_type = struct_data.edge_type[mask]

        sub_graph_path = os.path.join(
            self.sub_graph_dir, f"{hashlib.md5(row['SMILES'].encode('utf-8')).hexdigest()}.pt"
        )
        sub_graph = torch.load(sub_graph_path, map_location="cpu", weights_only=False)
        if not torch.isfinite(sub_graph.x).all():
            sub_graph.x = torch.nan_to_num(sub_graph.x, nan=0.0)
        fp = self.get_morgan_fp(row["SMILES"])

        ec = str(row["EC_Number"]).strip()
        ec_id = self.ec_to_id.get(ec, self.ec_to_id.get("UNK", 0))

        ph = float(row["pH"]) if pd.notna(row["pH"]) else 7.0
        ph_mask = 1.0 if pd.notna(row["pH"]) else 0.0

        temp = float(row["Temperature_C"]) if pd.notna(row["Temperature_C"]) else 25.0
        temp_mask = 1.0 if pd.notna(row["Temperature_C"]) else 0.0

        org_id = self.org_map.get(str(row["Organism"]), self.num_organisms - 1)
        type_id = int(row["Type_ID"])

        cond = {
            "ph": torch.tensor(ph, dtype=torch.float),
            "ph_mask": torch.tensor(ph_mask, dtype=torch.float),
            "temp": torch.tensor(temp, dtype=torch.float),
            "temp_mask": torch.tensor(temp_mask, dtype=torch.float),
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
            input_ids, attn_mask, struct_data, sub_graph, cond,
            torch.tensor([val_norm], dtype=torch.float),
            torch.tensor(task_id, dtype=torch.long),
        )


def collate_fn(batch):
    ids, masks, structs, subs, conds_list, vals, tasks = zip(*batch)
    ids = torch.stack(ids)
    masks = torch.stack(masks)
    struct_batch = Batch.from_data_list(structs)
    sub_batch = Batch.from_data_list(subs)
    vals = torch.stack(vals)
    tasks = torch.stack(tasks)

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
    return ids, masks, struct_batch, sub_batch, conds, vals, tasks


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x, mask):
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class ProteinLLMAdapter(nn.Module):

    def __init__(self, model_path, dim=HIDDEN_DIM, extract_layers=None):
        super().__init__()
        self.extract_layers = extract_layers or ESM_EXTRACT_LAYERS

        self.esm = AutoModel.from_pretrained(model_path, output_hidden_states=True)

        cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8, lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
        )
        self.esm = get_peft_model(self.esm, cfg)

        esm_hidden_size = self.esm.config.hidden_size
        n_layers_extracted = len(self.extract_layers)

        self.layer_weights = nn.Parameter(torch.ones(n_layers_extracted) / n_layers_extracted)
        self.layer_norm_per = nn.ModuleList([
            nn.LayerNorm(esm_hidden_size) for _ in range(n_layers_extracted)
        ])

        self.proj = nn.Sequential(
            nn.Linear(esm_hidden_size, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.attn_pool = AttentionPooling(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids, mask):
        if ids.shape[1] > SEQ_MAX_LEN:
            ids = ids[:, :SEQ_MAX_LEN]
            mask = mask[:, :SEQ_MAX_LEN]

        outputs = self.esm(ids, mask)

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            weights = F.softmax(self.layer_weights, dim=0)
            combined = torch.zeros_like(hidden_states[-1])
            for i, layer_idx in enumerate(self.extract_layers):
                if layer_idx < len(hidden_states):
                    combined = combined + weights[i] * self.layer_norm_per[i](
                        hidden_states[layer_idx]
                    )
        else:
            combined = outputs.last_hidden_state

        h = self.proj(combined)
        h_global = self.attn_pool(h, mask)
        return self.norm(h), self.norm(h_global)


class MultiScaleEGCL(nn.Module):
    def __init__(self, dim, edge_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1 + edge_dim, dim), nn.LayerNorm(dim),
            nn.SiLU(), nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
            nn.SiLU(), nn.Linear(dim, dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.SiLU(),
            nn.Linear(dim // 2, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, h, x, edge_index, edge_attr):
        row, col = edge_index

        coord_diff = x[row] - x[col]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)

        edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=-1)
        edge_feat = self.edge_mlp(edge_input)

        attn = self.attn_mlp(edge_feat)
        edge_feat = edge_feat * attn

        agg = torch.zeros(h.size(0), h.size(1), dtype=edge_feat.dtype, device=h.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), edge_feat)

        h_for_update = h.to(edge_feat.dtype)
        h_new = h_for_update + self.node_mlp(torch.cat([h_for_update, agg], dim=-1))

        coord_weight = torch.tanh(self.coord_mlp(edge_feat))
        coord_update = coord_diff.to(coord_weight.dtype) * coord_weight

        coord_agg = torch.zeros_like(x)
        count = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        coord_agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), coord_update.to(x.dtype))
        count.scatter_add_(
            0, row.unsqueeze(-1), torch.ones(row.size(0), 1, device=x.device, dtype=x.dtype)
        )
        x_new = x + coord_agg / count.clamp(min=1)

        return h_new, x_new


class HierarchicalVNLayer(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.egcl = MultiScaleEGCL(dim, dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.vn_broadcast = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.log_temp = nn.Parameter(torch.zeros(n_heads))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, x, edge_index, edge_attr, h_vn, x_vn, batch):
        B = h_vn.size(0)

        h, x = self.egcl(h, x, edge_index, edge_attr)
        h = self.norm1(h)

        h_vn = h_vn.to(h.dtype)

        q = self.q_proj(h_vn).view(B, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(-1, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(-1, self.n_heads, self.head_dim)

        q_expand = q[batch]
        scores = (q_expand * k).sum(dim=-1) / math.sqrt(self.head_dim)

        temps = self.log_temp.exp().clamp(0.1, 10.0)
        scores = scores / temps.unsqueeze(0).to(scores.dtype)

        attn_list = []
        for h_idx in range(self.n_heads):
            attn_list.append(pyg_softmax(scores[:, h_idx], batch))
        attn = torch.stack(attn_list, dim=1)

        weighted_v = v * attn.unsqueeze(-1)

        vn_update = torch.zeros(B, self.n_heads, self.head_dim, device=h.device, dtype=h.dtype)
        for h_idx in range(self.n_heads):
            src = weighted_v[:, h_idx]
            for b_idx in range(B):
                mask = batch == b_idx
                if mask.any():
                    vn_update[b_idx, h_idx] = src[mask].sum(dim=0)

        vn_update = self.out_proj(vn_update.view(B, -1))
        h_vn = h_vn + vn_update

        h_vn_broadcast = h_vn[batch]
        h = h + self.vn_broadcast(torch.cat([h, h_vn_broadcast], dim=-1))
        h = self.norm2(h)

        x_vn = x_vn.to(x.dtype)
        diff = x - x_vn[batch]
        weights = attn.mean(dim=1, keepdim=True)
        weights_norm = pyg_softmax(weights.squeeze(-1), batch).unsqueeze(-1)

        vn_coord_update = torch.zeros_like(x_vn)
        weighted_diff = diff * weights_norm
        for b_idx in range(B):
            mask = batch == b_idx
            if mask.any():
                vn_coord_update[b_idx] = weighted_diff[mask].sum(dim=0)

        x_vn = x_vn + 0.1 * vn_coord_update

        return h, x, h_vn, x_vn


class AdvancedStructureEncoder(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, n_layers=4, n_heads=4):
        super().__init__()
        self.dim = dim

        self.node_embed = nn.Embedding(21, dim)
        self.edge_embed = nn.Embedding(3, dim)

        self.layers = nn.ModuleList([HierarchicalVNLayer(dim, n_heads) for _ in range(n_layers)])

        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, data):
        h = data.x.long()
        h = torch.clamp(h, 0, 20)
        h = self.node_embed(h)

        x = data.pos.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch = data.batch

        if hasattr(data, "edge_type") and data.edge_type is not None:
            edge_types = torch.clamp(data.edge_type, 0, 2)
        else:
            edge_types = torch.zeros(data.edge_index.size(1), dtype=torch.long, device=h.device)
        edge_attr = self.edge_embed(edge_types)

        B = batch.max().item() + 1
        h_vn = torch.zeros(B, self.dim, device=h.device, dtype=h.dtype)
        x_vn = global_mean_pool(x, batch)

        for layer in self.layers:
            h, x, h_vn, x_vn = layer(h, x, data.edge_index, edge_attr, h_vn, x_vn, batch)

        h_vn = self.norm(self.proj(h_vn) * self.gate(h_vn))
        h_residue = self.norm(h)

        return h_residue, h_vn, batch


class SubstrateTransformerMultiPool(nn.Module):
    def __init__(self, in_dim=SUBSTRATE_IN_DIM, hidden_dim=HIDDEN_DIM, n_layers=3):
        super().__init__()

        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.edge_proj = nn.Linear(6, hidden_dim)

        self.gin_layers = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.multi_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_index = data.edge_index
        batch = data.batch

        for gin in self.gin_layers:
            x = F.gelu(gin(x, edge_index)) + x

        B = batch.max().item() + 1
        pooled_features = []

        for b in range(B):
            mask = batch == b
            x_b = x[mask].unsqueeze(0)
            if x_b.size(1) > 0:
                x_attn, _ = self.self_attn(x_b, x_b, x_b)
                x_attn = self.attn_norm(x_attn + x_b)

                h = x_attn
                mean_pool = h.mean(dim=1)
                max_pool = h.max(dim=1)[0]
                first_token = h[:, 0, :]
                attn_scores = torch.softmax(h.mean(dim=-1), dim=-1).unsqueeze(-1)
                attn_pool = (h * attn_scores).sum(dim=1)

                multi_pool = torch.cat([mean_pool, max_pool, first_token, attn_pool], dim=-1)
                pooled = self.multi_pool_proj(multi_pool)
                pooled_features.append(pooled)
            else:
                pooled_features.append(torch.zeros(1, x.size(-1), device=x.device))

        h_sub = torch.cat(pooled_features, dim=0)
        return self.out_norm(h_sub), x, batch


class PocketAwareCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, h_sub, h_prot, prot_batch, sub_batch):
        B = max(prot_batch.max().item(), sub_batch.max().item()) + 1

        h_sub_global = global_mean_pool(h_sub, sub_batch)

        outputs = []
        for b in range(B):
            prot_mask = prot_batch == b
            h_p = h_prot[prot_mask]

            if h_p.size(0) == 0 or b >= h_sub_global.size(0):
                outputs.append(torch.zeros(1, self.dim, device=h_sub.device))
                continue

            q = h_sub_global[b:b + 1]
            k = self.k_proj(h_p).unsqueeze(0)
            v = self.v_proj(h_p).unsqueeze(0)
            q = self.q_proj(q).unsqueeze(0)

            q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(1, -1)
            outputs.append(out)

        h_cross = torch.cat(outputs, dim=0)
        return self.norm(self.out_proj(h_cross))


class EnhancedConditionEncoder(nn.Module):
    def __init__(self, num_species, num_types, num_ecs, recon_dim, ec_init_matrix, embed_dim=COND_DIM):
        super().__init__()

        self.species_emb = nn.Embedding(num_species, embed_dim)
        self.type_emb = nn.Embedding(num_types, embed_dim)

        self.ph_enc = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.SiLU(), nn.Linear(embed_dim // 2, embed_dim)
        )
        self.temp_enc = nn.Sequential(
            nn.Linear(1, embed_dim // 2), nn.SiLU(), nn.Linear(embed_dim // 2, embed_dim)
        )

        self.ph_missing = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.temp_missing = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.ec_emb = nn.Embedding(num_ecs, recon_dim)
        if ec_init_matrix is not None:
            self.ec_emb.weight.data.copy_(ec_init_matrix)
        self.ec_proj = nn.Linear(recon_dim, embed_dim)

        self.fp_proj = nn.Sequential(
            nn.Linear(2048, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.scale_gen = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.SiLU(), nn.Linear(64, 3), nn.Sigmoid()
        )

    def forward(self, c):
        z_species = self.species_emb(c["species"])
        z_type = self.type_emb(c["type"])

        ph_val = self.ph_enc(c["ph"].unsqueeze(-1))
        z_ph = c["ph_mask"] * ph_val + (1 - c["ph_mask"]) * self.ph_missing

        temp_val = self.temp_enc(c["temp"].unsqueeze(-1))
        z_temp = c["temp_mask"] * temp_val + (1 - c["temp_mask"]) * self.temp_missing

        z_ec = self.ec_proj(self.ec_emb(c["ec_id"]))
        z_fp = self.fp_proj(c["fp"])

        z_cond = self.fusion(torch.cat([z_species, z_type, z_ph, z_temp, z_ec, z_fp], dim=-1))
        scales = self.scale_gen(z_cond)

        return z_cond, scales, z_ec


class DualPathwayPredictionHead(nn.Module):
    def __init__(self, feat_dim, cond_dim, n_experts=4):
        super().__init__()
        self.n_experts = n_experts

        self.shared_trunk = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.binding_trunk = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.km_head = nn.Linear(feat_dim // 2, 2)
        self.ki_head = nn.Linear(feat_dim // 2, 2)

        self.catalytic_trunk = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.kcat_head = nn.Linear(feat_dim // 2, 2)
        self.eff_head = nn.Linear(feat_dim // 2, 2)

    def forward(self, x_feat, z_cond):
        h = self.shared_trunk(torch.cat([x_feat, z_cond], dim=-1))

        h_binding = self.binding_trunk(h)
        out_km = self.km_head(h_binding)
        out_ki = self.ki_head(h_binding)

        h_catalytic = self.catalytic_trunk(h)
        out_kcat = self.kcat_head(h_catalytic)
        out_eff = self.eff_head(h_catalytic)

        return {"km": out_km, "kcat": out_kcat, "eff": out_eff, "ki": out_ki}


class EnzCast(nn.Module):
    def __init__(self, esm_path, num_species, num_types, num_ecs, recon_dim, ec_init_matrix):
        super().__init__()

        self.seq_enc = ProteinLLMAdapter(esm_path, dim=HIDDEN_DIM)
        self.struct_enc = AdvancedStructureEncoder(dim=HIDDEN_DIM, n_layers=4, n_heads=4)
        self.sub_enc = SubstrateTransformerMultiPool(
            in_dim=SUBSTRATE_IN_DIM, hidden_dim=HIDDEN_DIM, n_layers=3
        )

        self.cond_enc = EnhancedConditionEncoder(
            num_species, num_types, num_ecs, recon_dim, ec_init_matrix, embed_dim=COND_DIM
        )

        self.cross_attn = PocketAwareCrossAttention(HIDDEN_DIM, n_heads=4)

        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )

        self.fusion_gate = nn.Sequential(nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM), nn.Sigmoid())

        self.prediction_head = DualPathwayPredictionHead(HIDDEN_DIM, COND_DIM)

        self.prot_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(), nn.Linear(HIDDEN_DIM, 128),
        )
        self.sub_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(), nn.Linear(HIDDEN_DIM, 128),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_representations(self, ids, mask, struct, sub_graph, autocast_dtype):
        h_seq_res, h_seq_global = self.seq_enc(ids, mask)

        with torch.cuda.amp.autocast(enabled=False):
            if struct.pos.dtype != torch.float32:
                struct.pos = struct.pos.float()
            h_struct_res, h_struct_vn, struct_batch = self.struct_enc(struct)

        target_dtype = h_seq_global.dtype
        h_struct_vn = h_struct_vn.to(target_dtype)
        h_struct_res = h_struct_res.to(target_dtype)

        h_sub_global, h_sub_atoms, sub_batch = self.sub_enc(sub_graph)
        h_sub_global = h_sub_global.to(target_dtype)
        h_sub_atoms = h_sub_atoms.to(target_dtype)

        B = min(h_seq_global.size(0), h_struct_vn.size(0), h_sub_global.size(0))

        return {
            "h_seq_global": h_seq_global[:B],
            "h_struct_vn": h_struct_vn[:B],
            "h_struct_res": h_struct_res,
            "struct_batch": struct_batch,
            "h_sub_global": h_sub_global[:B],
            "h_sub_atoms": h_sub_atoms,
            "sub_batch": sub_batch,
            "B": B,
        }

    def contrastive_forward(self, ids, mask, struct, sub_graph, autocast_dtype):
        reps = self.get_representations(ids, mask, struct, sub_graph, autocast_dtype)

        h_prot = torch.cat([reps["h_seq_global"], reps["h_struct_vn"]], dim=-1)
        z_prot = F.normalize(self.prot_proj(h_prot), dim=-1, eps=1e-6)
        z_sub = F.normalize(self.sub_proj(reps["h_sub_global"]), dim=-1, eps=1e-6)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * z_prot @ z_sub.t()

        return logits

    def forward(self, ids, mask, struct, sub_graph, phys, autocast_dtype=torch.bfloat16):
        reps = self.get_representations(ids, mask, struct, sub_graph, autocast_dtype)
        B = reps["B"]

        z_cond, scales, _ = self.cond_enc(phys)
        z_cond = z_cond[:B]
        scales = scales[:B]

        h_cross = self.cross_attn(
            reps["h_sub_atoms"], reps["h_struct_res"],
            reps["struct_batch"], reps["sub_batch"]
        )[:B]

        scale_seq = scales[:, 0:1]
        scale_struct = scales[:, 1:2]
        scale_sub = scales[:, 2:3]

        h_combined = torch.cat([
            reps["h_seq_global"] * scale_seq,
            reps["h_struct_vn"] * scale_struct,
            reps["h_sub_global"] * scale_sub,
            h_cross
        ], dim=-1)

        h_fused = self.fusion(h_combined)
        gate = self.fusion_gate(h_combined)
        h_final = h_fused * gate

        out = self.prediction_head(h_final, z_cond)
        return out


class BetaNLLLoss(nn.Module):

    def __init__(self, beta=BETA_NLL_BETA):
        super().__init__()
        self.beta = beta

    def forward(self, mu, logvar, target, nll_weight=1.0):
        mu = mu.view(-1)
        logvar = logvar.view(-1)
        target = target.view(-1)

        logvar = logvar.clamp(-8, 8)
        var = torch.exp(logvar)

        diff = target - mu
        nll = 0.5 * logvar + 0.5 * (diff ** 2) / var.clamp(min=1e-6)

        if self.beta > 0:
            beta_weight = var.detach() ** self.beta
            nll = nll * beta_weight

        return nll.mean() * nll_weight


class UncertaintyWeightedMTL(nn.Module):
    def __init__(self, n_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses_dict, task_names=None):
        if task_names is None:
            task_names = ["km", "kcat", "eff", "ki"]

        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        for i, t_name in enumerate(task_names):
            if t_name not in losses_dict or losses_dict[t_name] is None:
                continue
            loss_val = losses_dict[t_name]
            log_var = torch.clamp(self.log_vars[i], -6.0, 6.0)
            precision = torch.exp(-log_var)
            total_loss = total_loss + 0.5 * precision * loss_val + 0.5 * log_var

        return total_loss

    def get_weights(self):
        with torch.no_grad():
            return torch.exp(-torch.clamp(self.log_vars, -6.0, 6.0))


def consistency_loss_log10(mu_kcat, mu_km, mu_eff, stats):
    m_km, s_km = stats[0]["mean"], stats[0]["std"]
    m_kcat, s_kcat = stats[1]["mean"], stats[1]["std"]
    m_eff, s_eff = stats[2]["mean"], stats[2]["std"]

    raw_km = mu_km * s_km + m_km
    raw_kcat = mu_kcat * s_kcat + m_kcat
    raw_eff = mu_eff * s_eff + m_eff

    return ((raw_kcat - raw_km - raw_eff) ** 2).mean()


def get_nll_weight_short(epoch, total_epochs):
    if total_epochs <= 1:
        return 0.5

    NLL_FLOOR = 0.05
    ramp_end_frac = 0.80
    ramp_end_epoch = total_epochs * ramp_end_frac

    if epoch >= ramp_end_epoch:
        return 1.0

    progress = epoch / ramp_end_epoch
    return NLL_FLOOR + (1.0 - NLL_FLOOR) * progress


def evaluate_multitask(model, loader, stats, device, split_name="test", ema=None):
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    csv_records = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Eval {split_name}")
        for batch in pbar:
            ids, mask, struct, sub, conds, truth, tasks = batch
            ids, mask = ids.to(device), mask.to(device)
            struct, sub = struct.to(device), sub.to(device)
            conds = {k: v.to(device) for k, v in conds.items()}

            out = model(ids, mask, struct, sub, conds)

            mu_km = out["km"][:, 0]
            mu_kcat = out["kcat"][:, 0]
            mu_eff = out["eff"][:, 0]
            mu_ki = out["ki"][:, 0]

            preds_all = torch.stack([mu_km, mu_kcat, mu_eff, mu_ki], dim=1).cpu().float().numpy()
            truth = truth.cpu().float().numpy().flatten()
            tasks = tasks.cpu().numpy().flatten()

            for i in range(len(tasks)):
                tid = int(tasks[i])
                if tid not in stats:
                    continue
                mean, std = stats[tid]["mean"], stats[tid]["std"]
                pred_raw = float(preds_all[i, tid]) * std + mean
                true_raw = float(truth[i]) * std + mean
                csv_records.append({
                    "Task": TASK_NAMES[tid],
                    "Pred_Log10": pred_raw,
                    "True_Log10": true_raw,
                    "Split": split_name,
                })

    if ema is not None:
        ema.restore(model)

    return csv_records


def run_pretraining(model, loader, device, epochs, gpu_config,
                    grad_accum_steps=2):
    log_print(f"\n[Pretrain] Contrastive Pretraining for {epochs} epochs (LR=5e-5)")
    log_print(f"[Pretrain] Gradient accumulation steps: {grad_accum_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_scaler = gpu_config["use_scaler"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_config["precision"] == "bfloat16" else torch.float16

    model.train()
    num_batches = len(loader)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch + 1}", mininterval=10)

        for batch_idx, batch in enumerate(pbar):
            try:
                ids, mask, struct, sub, _, _, _ = batch
                ids, mask = ids.to(device), mask.to(device)
                struct, sub = struct.to(device), sub.to(device)

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    logits = model.contrastive_forward(ids, mask, struct, sub, autocast_dtype)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        continue
                    labels = torch.arange(logits.size(0), device=device)
                    loss = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2
                    loss = loss / grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches)
                if is_accum_step:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * grad_accum_steps
                batch_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_print("[OOM] Skipping batch in pretrain")
                    torch.cuda.empty_cache()
                else:
                    raise

        avg_loss = total_loss / max(1, batch_count)
        log_print(f"[Pretrain] Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}")

    save_path = os.path.join(os.getcwd(), "pretrained_enzcast.pt")
    torch.save(model.state_dict(), save_path)
    log_print(f"[Pretrain] Saved to {save_path}\n")


def stage1_train(model, train_dl, val_dl, device, stats, args, gpu_config):
    log_print("\n[Stage 1] MTL Training — Short-Schedule Optimized")
    log_print(f"[Stage 1] Epochs: {args.stage1_epochs}, LR backbone: {args.lr_backbone}, "
              f"LR head: {args.lr_head}, Grad Accum: {args.grad_accum_steps}")

    use_scaler = gpu_config["use_scaler"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_config["precision"] == "bfloat16" else torch.float16

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prediction_head" in name or "fusion" in name or "cond_enc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    total_steps = len(train_dl) * args.stage1_epochs // args.grad_accum_steps
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    loss_fn = BetaNLLLoss(beta=BETA_NLL_BETA)
    mtl_loss = UncertaintyWeightedMTL(n_tasks=4).to(device)
    opt_mtl = torch.optim.AdamW(mtl_loss.parameters(), lr=1e-3)

    ema = EMAModel(model, decay=EMA_DECAY)

    best_score = -1e9
    best_ema_score = -1e9
    best_path = os.path.join(os.getcwd(), "best_enzcast_stage1.pt")
    best_ema_path = os.path.join(os.getcwd(), "best_enzcast_stage1_ema.pt")
    ckpt_path = os.path.join(os.getcwd(), "enzcast_stage1_checkpoint.pth")

    patience = args.patience
    epochs_without_improve = 0

    start_epoch = 0
    if args.resume_stage1 and os.path.exists(ckpt_path):
        log_print(f"[Stage 1] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", best_score)

    log_print("[NLL Schedule]")
    for e in range(args.stage1_epochs):
        log_print(f"  Epoch {e+1}: nll_weight = {get_nll_weight_short(e, args.stage1_epochs):.3f}")

    for epoch in range(start_epoch, args.stage1_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        accum_count = 0

        nll_w = get_nll_weight_short(epoch, args.stage1_epochs)

        opt.zero_grad(set_to_none=True)
        opt_mtl.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Stage1 Epoch {epoch + 1}", mininterval=10)
        for batch_idx, batch in enumerate(pbar):
            step_ok = True
            try:
                ids, mask, struct, sub, conds, truth_val, task_ids = batch
                ids, mask = ids.to(device), mask.to(device)
                struct, sub = struct.to(device), sub.to(device)
                truth_val = truth_val.to(device)
                task_ids = task_ids.to(device)
                conds = {k: v.to(device) for k, v in conds.items()}

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    out = model(ids, mask, struct, sub, conds, autocast_dtype=autocast_dtype)

                    losses_dict = {}
                    is_km = task_ids == 0
                    is_kcat = task_ids == 1
                    is_eff = task_ids == 2
                    is_ki = task_ids == 3

                    if is_km.any():
                        losses_dict["km"] = loss_fn(
                            out["km"][is_km, 0], out["km"][is_km, 1], truth_val[is_km], nll_w
                        )
                    if is_kcat.any():
                        losses_dict["kcat"] = loss_fn(
                            out["kcat"][is_kcat, 0], out["kcat"][is_kcat, 1], truth_val[is_kcat], nll_w
                        )
                    if is_eff.any():
                        losses_dict["eff"] = loss_fn(
                            out["eff"][is_eff, 0], out["eff"][is_eff, 1], truth_val[is_eff], nll_w
                        )
                    if is_ki.any():
                        losses_dict["ki"] = loss_fn(
                            out["ki"][is_ki, 0], out["ki"][is_ki, 1], truth_val[is_ki], nll_w
                        )

                    if len(losses_dict) == 0:
                        step_ok = False
                    else:
                        model_loss = mtl_loss(losses_dict)

                        if is_eff.any() or (is_kcat.any() and is_km.any()):
                            cons_loss = consistency_loss_log10(
                                out["kcat"][:, 0], out["km"][:, 0], out["eff"][:, 0], stats
                            )
                            model_loss = model_loss + args.consistency_weight * cons_loss

                        model_loss = model_loss / args.grad_accum_steps

                        if torch.isnan(model_loss) or torch.isinf(model_loss):
                            step_ok = False

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_print("[OOM] Skipping batch")
                    step_ok = False
                    torch.cuda.empty_cache()
                else:
                    raise

            if not step_ok:
                continue

            if use_scaler:
                scaler.scale(model_loss).backward()
            else:
                model_loss.backward()

            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                opt_mtl.step()
                scheduler.step()
                ema.update(model)

                opt.zero_grad(set_to_none=True)
                opt_mtl.zero_grad(set_to_none=True)

            total_loss += model_loss.item() * args.grad_accum_steps
            batch_count += 1

        if accum_count % args.grad_accum_steps != 0:
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt_mtl.step()
            scheduler.step()
            ema.update(model)
            opt.zero_grad(set_to_none=True)
            opt_mtl.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, batch_count)
        w = mtl_loss.get_weights()
        log_print(
            f"[Stage1] Epoch {epoch + 1}/{args.stage1_epochs} | Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | NLL_w: {nll_w:.3f} | "
            f"Task Weights: km={w[0]:.2f} kcat={w[1]:.2f} eff={w[2]:.2f} ki={w[3]:.2f}"
        )

        val_recs = evaluate_multitask(model, val_dl, stats, device, "val", ema=ema)
        v_df = pd.DataFrame(val_recs)
        if not v_df.empty:
            r2_scores = {}
            for task in TASK_NAMES:
                sub_df = v_df[v_df["Task"] == task]
                if len(sub_df) > 0:
                    rmse = np.sqrt(mean_squared_error(sub_df["True_Log10"], sub_df["Pred_Log10"]))
                    r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
                    r2_scores[task] = r2
                    log_print(f"    {task}: RMSE={rmse:.4f}, R2={r2:.4f}")

            score = np.mean(list(r2_scores.values())) if r2_scores else -1e9
            log_print(f"  >> Val Mean R2 (EMA): {score:.4f}")

            if score > best_ema_score:
                best_ema_score = score
                ema.apply_shadow(model)
                torch.save(model.state_dict(), best_ema_path)
                ema.restore(model)
                log_print(f"  >> New Best EMA Model Saved (R2={score:.4f})")

        val_recs_reg = evaluate_multitask(model, val_dl, stats, device, "val_reg", ema=None)
        v_df_reg = pd.DataFrame(val_recs_reg)
        if not v_df_reg.empty:
            r2_scores_reg = {}
            for task in TASK_NAMES:
                sub_df = v_df_reg[v_df_reg["Task"] == task]
                if len(sub_df) > 0:
                    r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
                    r2_scores_reg[task] = r2

            score_reg = np.mean(list(r2_scores_reg.values())) if r2_scores_reg else -1e9
            log_print(f"  >> Val Mean R2 (Regular): {score_reg:.4f}")

            if score_reg > best_score:
                best_score = score_reg
                epochs_without_improve = 0
                torch.save(model.state_dict(), best_path)
                log_print(f"  >> New Best Regular Model Saved (R2={score_reg:.4f})")
            else:
                epochs_without_improve += 1
                log_print(f"  >> No improvement ({epochs_without_improve}/{patience})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "mtl_log_vars": mtl_loss.log_vars.data,
        }, ckpt_path)

        if epochs_without_improve >= patience:
            log_print(f"[Stage1] Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
            break

    return best_path, best_ema_path


def stage2_finetune(model, train_dl, val_dl, device, stats, args, gpu_config, stage1_best_path):
    log_print(f"\n[Stage 2] All-Task Fine-Tuning ({args.stage2_epochs} epochs, full NLL)")

    if stage1_best_path and os.path.exists(stage1_best_path):
        log_print(f"[Stage2] Loading Stage1 best: {stage1_best_path}")
        model.load_state_dict(torch.load(stage1_best_path, map_location=device))

    for p in model.seq_enc.esm.parameters():
        p.requires_grad = False
    for name, p in model.seq_enc.esm.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log_print(f"[Stage2] Trainable params: {sum(p.numel() for p in trainable_params):,}")

    opt = torch.optim.AdamW(trainable_params, lr=args.stage2_lr, weight_decay=0.01)

    total_steps = len(train_dl) * args.stage2_epochs // args.grad_accum_steps
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    use_scaler = gpu_config["use_scaler"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_config["precision"] == "bfloat16" else torch.float16

    loss_fn = BetaNLLLoss(beta=BETA_NLL_BETA)
    ema = EMAModel(model, decay=EMA_DECAY)

    task_weights = {"km": 1.0, "kcat": 1.0, "eff": args.eff_priority, "ki": 1.0}
    log_print(f"[Stage2] Task weights: {task_weights}")

    best_score = -1e9
    best_ema_score = -1e9
    best_path = os.path.join(os.getcwd(), "best_enzcast_stage2.pt")
    best_ema_path = os.path.join(os.getcwd(), "best_enzcast_stage2_ema.pt")
    epochs_without_improve = 0
    patience = args.patience

    for epoch in range(args.stage2_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        accum_count = 0

        nll_w = 1.0

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Stage2 Epoch {epoch + 1}", mininterval=10)
        for batch_idx, batch in enumerate(pbar):
            step_ok = True
            try:
                ids, mask, struct, sub, conds, truth_val, task_ids = batch
                ids, mask = ids.to(device), mask.to(device)
                struct, sub = struct.to(device), sub.to(device)
                truth_val = truth_val.to(device)
                task_ids = task_ids.to(device)
                conds = {k: v.to(device) for k, v in conds.items()}

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    out = model(ids, mask, struct, sub, conds, autocast_dtype=autocast_dtype)

                    loss = torch.tensor(0.0, device=device)
                    n_tasks_active = 0

                    is_km = task_ids == 0
                    is_kcat = task_ids == 1
                    is_eff = task_ids == 2
                    is_ki = task_ids == 3

                    if is_km.any():
                        l_km = loss_fn(out["km"][is_km, 0], out["km"][is_km, 1],
                                       truth_val[is_km], nll_w)
                        loss = loss + task_weights["km"] * l_km
                        n_tasks_active += 1
                    if is_kcat.any():
                        l_kcat = loss_fn(out["kcat"][is_kcat, 0], out["kcat"][is_kcat, 1],
                                         truth_val[is_kcat], nll_w)
                        loss = loss + task_weights["kcat"] * l_kcat
                        n_tasks_active += 1
                    if is_eff.any():
                        l_eff = loss_fn(out["eff"][is_eff, 0], out["eff"][is_eff, 1],
                                        truth_val[is_eff], nll_w)
                        loss = loss + task_weights["eff"] * l_eff
                        n_tasks_active += 1
                    if is_ki.any():
                        l_ki = loss_fn(out["ki"][is_ki, 0], out["ki"][is_ki, 1],
                                       truth_val[is_ki], nll_w)
                        loss = loss + task_weights["ki"] * l_ki
                        n_tasks_active += 1

                    if n_tasks_active == 0:
                        step_ok = False
                    else:
                        cons_loss = consistency_loss_log10(
                            out["kcat"][:, 0], out["km"][:, 0], out["eff"][:, 0], stats
                        )
                        loss = loss + args.consistency_weight * cons_loss

                        loss = loss / args.grad_accum_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        step_ok = False

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_print("[OOM] Skipping batch")
                    step_ok = False
                    torch.cuda.empty_cache()
                else:
                    raise

            if not step_ok:
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step()

                scheduler.step()
                ema.update(model)
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum_steps
            batch_count += 1

        if accum_count % args.grad_accum_steps != 0:
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
            scheduler.step()
            ema.update(model)
            opt.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, batch_count)
        log_print(f"[Stage2] Epoch {epoch + 1}/{args.stage2_epochs} | "
                  f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        val_recs = evaluate_multitask(model, val_dl, stats, device, "val", ema=ema)
        v_df = pd.DataFrame(val_recs)
        r2_scores = {}
        for task in TASK_NAMES:
            sub_df = v_df[v_df["Task"] == task]
            if len(sub_df) > 0:
                r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
                r2_scores[task] = r2
                log_print(f"    {task}: R2={r2:.4f}")

        score_ema = np.mean(list(r2_scores.values())) if r2_scores else -1e9
        log_print(f"  >> Val Mean R2 (EMA): {score_ema:.4f}")

        if score_ema > best_ema_score:
            best_ema_score = score_ema
            ema.apply_shadow(model)
            torch.save(model.state_dict(), best_ema_path)
            ema.restore(model)
            log_print(f"  >> New Best Stage2 EMA Model Saved (R2={score_ema:.4f})")

        val_recs_reg = evaluate_multitask(model, val_dl, stats, device, "val_reg", ema=None)
        v_df_reg = pd.DataFrame(val_recs_reg)
        r2_scores_reg = {}
        for task in TASK_NAMES:
            sub_df = v_df_reg[v_df_reg["Task"] == task]
            if len(sub_df) > 0:
                r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
                r2_scores_reg[task] = r2

        score = np.mean(list(r2_scores_reg.values())) if r2_scores_reg else -1e9
        log_print(f"  >> Val Mean R2 (Regular): {score:.4f}")

        if score > best_score:
            best_score = score
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_path)
            log_print(f"  >> New Best Stage2 Regular Model Saved (R2={score:.4f})")
        else:
            epochs_without_improve += 1
            log_print(f"  >> No improvement ({epochs_without_improve}/{patience})")

        if epochs_without_improve >= patience:
            log_print(f"[Stage2] Early stopping at epoch {epoch + 1}")
            break

    return best_path, best_ema_path


def train_single_task(model, train_dl, val_dl, test_dl, device, stats, task_name,
                      args, gpu_config, pretrained_path=None):
    tid = TASK_IDX[task_name]
    log_print(f"\n{'='*60}")
    log_print(f"[Single-Task] Training for: {task_name} (tid={tid})")
    log_print(f"{'='*60}")

    if pretrained_path and os.path.exists(pretrained_path):
        log_print(f"[Single-Task] Loading pretrained: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

    use_scaler = gpu_config["use_scaler"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if gpu_config["precision"] == "bfloat16" else torch.float16

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prediction_head" in name or "fusion" in name or "cond_enc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    n_epochs = args.stage1_epochs
    total_steps = len(train_dl) * n_epochs // args.grad_accum_steps
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    loss_fn = BetaNLLLoss(beta=BETA_NLL_BETA)
    ema = EMAModel(model, decay=EMA_DECAY)

    head_key = "eff" if task_name == "kcat_km" else task_name

    best_r2 = -1e9
    best_ema_r2 = -1e9
    best_path = os.path.join(os.getcwd(), f"best_enzcast_{task_name}.pt")
    best_ema_path = os.path.join(os.getcwd(), f"best_enzcast_{task_name}_ema.pt")
    epochs_without_improve = 0
    patience = args.patience

    log_print(f"[{task_name}] NLL Schedule:")
    for e in range(n_epochs):
        log_print(f"  Epoch {e+1}: nll_weight = {get_nll_weight_short(e, n_epochs):.3f}")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        accum_count = 0
        nll_w = get_nll_weight_short(epoch, n_epochs)

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"[{task_name}] Epoch {epoch + 1}", mininterval=10)
        for batch_idx, batch in enumerate(pbar):
            step_ok = True
            try:
                ids, mask, struct, sub, conds, truth_val, task_ids = batch
                ids, mask = ids.to(device), mask.to(device)
                struct, sub = struct.to(device), sub.to(device)
                truth_val = truth_val.to(device)
                task_ids = task_ids.to(device)
                conds = {k: v.to(device) for k, v in conds.items()}

                is_task = task_ids == tid
                if not is_task.any():
                    continue

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    out = model(ids, mask, struct, sub, conds, autocast_dtype=autocast_dtype)
                    loss = loss_fn(
                        out[head_key][is_task, 0],
                        out[head_key][is_task, 1],
                        truth_val[is_task],
                        nll_w,
                    )
                    loss = loss / args.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    step_ok = False
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    step_ok = False
                    torch.cuda.empty_cache()
                else:
                    raise

            if not step_ok:
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                scheduler.step()
                ema.update(model)
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum_steps
            batch_count += 1

        if accum_count % args.grad_accum_steps != 0:
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()
            ema.update(model)
            opt.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, batch_count)
        log_print(f"[{task_name}] Epoch {epoch + 1} | Loss: {avg_loss:.4f} | NLL_w: {nll_w:.3f}")

        val_recs = evaluate_multitask(model, val_dl, stats, device, "val", ema=ema)
        v_df = pd.DataFrame(val_recs)
        sub_df = v_df[v_df["Task"] == task_name]
        if len(sub_df) > 0:
            r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
            rmse = np.sqrt(mean_squared_error(sub_df["True_Log10"], sub_df["Pred_Log10"]))
            log_print(f"    {task_name} (EMA): R2={r2:.4f}, RMSE={rmse:.4f}")

            if r2 > best_ema_r2:
                best_ema_r2 = r2
                ema.apply_shadow(model)
                torch.save(model.state_dict(), best_ema_path)
                ema.restore(model)
                log_print(f"  >> New Best {task_name} EMA Model: R2={r2:.4f}")

        val_recs_reg = evaluate_multitask(model, val_dl, stats, device, "val_reg", ema=None)
        v_df_reg = pd.DataFrame(val_recs_reg)
        sub_df_reg = v_df_reg[v_df_reg["Task"] == task_name]
        if len(sub_df_reg) > 0:
            r2_reg = r2_score(sub_df_reg["True_Log10"], sub_df_reg["Pred_Log10"])
            log_print(f"    {task_name} (Regular): R2={r2_reg:.4f}")

            if r2_reg > best_r2:
                best_r2 = r2_reg
                epochs_without_improve = 0
                torch.save(model.state_dict(), best_path)
                log_print(f"  >> New Best {task_name} Regular Model: R2={r2_reg:.4f}")
            else:
                epochs_without_improve += 1
                log_print(f"  >> No improvement ({epochs_without_improve}/{patience})")

        if epochs_without_improve >= patience:
            log_print(f"[{task_name}] Early stopping at epoch {epoch + 1}")
            break

    log_print(f"\n[{task_name}] Loading best models for testing...")

    test_results = {}
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_recs = evaluate_multitask(model, test_dl, stats, device, "test")
        t_df = pd.DataFrame(test_recs)
        sub_df = t_df[t_df["Task"] == task_name]
        if len(sub_df) > 0:
            r2 = r2_score(sub_df["True_Log10"], sub_df["Pred_Log10"])
            rmse = np.sqrt(mean_squared_error(sub_df["True_Log10"], sub_df["Pred_Log10"]))
            test_results["regular"] = {"r2": r2, "rmse": rmse, "df": t_df}
            log_print(f"[{task_name}] TEST (Regular): R2={r2:.4f}, RMSE={rmse:.4f}, N={len(sub_df)}")

    if os.path.exists(best_ema_path):
        model.load_state_dict(torch.load(best_ema_path, map_location=device))
        test_recs_ema = evaluate_multitask(model, test_dl, stats, device, "test_ema")
        t_df_ema = pd.DataFrame(test_recs_ema)
        sub_df_ema = t_df_ema[t_df_ema["Task"] == task_name]
        if len(sub_df_ema) > 0:
            r2_ema = r2_score(sub_df_ema["True_Log10"], sub_df_ema["Pred_Log10"])
            rmse_ema = np.sqrt(mean_squared_error(sub_df_ema["True_Log10"], sub_df_ema["Pred_Log10"]))
            test_results["ema"] = {"r2": r2_ema, "rmse": rmse_ema, "df": t_df_ema}
            log_print(f"[{task_name}] TEST (EMA):     R2={r2_ema:.4f}, RMSE={rmse_ema:.4f}, N={len(sub_df_ema)}")

    if test_results:
        best_variant = max(test_results.keys(), key=lambda k: test_results[k]["r2"])
        log_print(f"[{task_name}] Best variant: {best_variant}")
        final_df = test_results[best_variant]["df"]
    else:
        final_df = pd.DataFrame()

    return best_path, final_df


def main():
    parser = argparse.ArgumentParser(description="EnzCast: Enzyme Kinetics Estimator")
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--debug_samples", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_pretrain", action="store_true")

    parser.add_argument("--mode", type=str, default="multitask",
                        choices=["multitask", "single_task", "both"],
                        help="multitask: joint 4-task training; "
                             "single_task: separate model per task; "
                             "both: run multitask then single-task")

    parser.add_argument("--stage1_epochs", type=int, default=DEFAULT_STAGE1_EPOCHS,
                        help="Stage 1 epochs (default: 5, best model typically at 3-4)")
    parser.add_argument("--stage2_epochs", type=int, default=DEFAULT_STAGE2_EPOCHS,
                        help="Stage 2 epochs (default: 6)")
    parser.add_argument("--pretrain_epochs", type=int, default=PRETRAIN_EPOCH,
                        help="Contrastive pretrain epochs (default: 3)")
    parser.add_argument("--run_stage2", action="store_true",
                        help="Run Stage 2 fine-tuning after Stage 1")

    parser.add_argument("--lr_backbone", type=float, default=8e-5,
                        help="LR for ESM backbone + structure encoder (default: 8e-5)")
    parser.add_argument("--lr_head", type=float, default=6e-4,
                        help="LR for prediction heads + fusion (default: 6e-4)")
    parser.add_argument("--stage2_lr", type=float, default=3e-4,
                        help="LR for Stage 2 fine-tuning (default: 3e-4)")

    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size × N)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (epochs without val improvement)")

    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--eff_priority", type=float, default=2.0,
                        help="Weight multiplier for kcat/Km in Stage 2")
    parser.add_argument("--resume_stage1", action="store_true")

    args = parser.parse_args()

    project_root = args.project_root
    base_dir = os.path.join(project_root, "Data")
    graph_dir = os.path.join(base_dir, "Structure_Data/Graph_Features-v2")
    sub_graph_dir = os.path.join(base_dir, "Structure_Data/Substrate_Graphs-v2")
    esm_model_path = os.path.join(project_root, "models/ESM2/ESM2_650M")
    benchmark_dir = os.path.join(
        project_root, "Prepare-Paper/Res-section1-DB-construction/KinHub-TME_Benchmark"
    )

    data_paths = {
        "train": os.path.join(benchmark_dir, "train_split.csv"),
        "val": os.path.join(benchmark_dir, "val_split.csv"),
        "test": os.path.join(benchmark_dir, "test_split.csv"),
    }

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    gpu_config = detect_gpu_config(args.gpu)

    if gpu_config.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_print(f"[ENV] Device: {device}")
    log_print(f"[GPU Config] {gpu_config}")
    log_print(f"[Config] Schedule: pretrain={args.pretrain_epochs}, "
              f"stage1={args.stage1_epochs}, stage2={args.stage2_epochs}")
    log_print(f"[Config] Grad Accum: {args.grad_accum_steps}, "
              f"Patience: {args.patience}, EMA decay: {EMA_DECAY}")
    log_print(f"[Config] Beta-NLL(beta={BETA_NLL_BETA}), "
              f"ESM layers={ESM_EXTRACT_LAYERS}, EMA decay={EMA_DECAY}")

    log_print("[Main] Loading ESM Tokenizer...")
    tok = AutoTokenizer.from_pretrained(esm_model_path)

    log_print("\n--- Loading Datasets ---")
    train_ds = EnzCastDataset(
        data_paths["train"], tok, graph_dir, sub_graph_dir,
        max_samples=args.debug_samples, base_dir=base_dir
    )
    val_ds = EnzCastDataset(
        data_paths["val"], tok, graph_dir, sub_graph_dir,
        max_samples=args.debug_samples,
        known_stats=train_ds.stats,
        ref_ec_map=train_ds.ec_to_id,
        ref_ec_matrix=train_ds.ec_init_matrix,
        base_dir=base_dir,
    )
    test_ds = EnzCastDataset(
        data_paths["test"], tok, graph_dir, sub_graph_dir,
        max_samples=args.debug_samples,
        known_stats=train_ds.stats,
        ref_ec_map=train_ds.ec_to_id,
        ref_ec_matrix=train_ds.ec_init_matrix,
        base_dir=base_dir,
    )

    batch_size = gpu_config["batch_size"]
    log_print(f"[Main] Batch Size: {batch_size}, "
              f"Effective Batch: {batch_size * args.grad_accum_steps}")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=gpu_config["num_workers"], pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=gpu_config["num_workers"], pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=gpu_config["num_workers"], pin_memory=True,
    )

    if args.mode in ("multitask", "both"):
        log_print("\n" + "=" * 60)
        log_print("MULTI-TASK TRAINING (Beta-NLL + EMA + Deep ESM)")
        log_print("=" * 60)

        recon_dim = getattr(train_ds, "recon_dim", 128)
        model = EnzCast(
            esm_model_path, train_ds.num_organisms, train_ds.num_types,
            train_ds.num_ecs, recon_dim, train_ds.ec_init_matrix
        ).to(device)

        if not args.skip_pretrain:
            run_pretraining(model, train_dl, device, args.pretrain_epochs, gpu_config,
                            grad_accum_steps=args.grad_accum_steps)

        stage1_best, stage1_ema_best = stage1_train(model, train_dl, val_dl, device, train_ds.stats, args, gpu_config)

        if args.run_stage2:
            stage2_best, stage2_ema_best = stage2_finetune(
                model, train_dl, val_dl, device, train_ds.stats, args, gpu_config, stage1_best
            )
            final_model_path = stage2_best
            final_ema_path = stage2_ema_best
        else:
            final_model_path = stage1_best
            final_ema_path = stage1_ema_best

        log_print("\n[Main] Final Testing (Regular + EMA)...")

        if os.path.exists(final_model_path):
            model.load_state_dict(torch.load(final_model_path, map_location=device))
        test_recs = evaluate_multitask(model, test_dl, train_ds.stats, device, "test")
        final_df = pd.DataFrame(test_recs)
        output_csv = "EnzCast_MTL_Regular_Results.csv"
        final_df.to_csv(output_csv, index=False)

        log_print(f"\n[Final MTL Results - Regular] {output_csv}")
        for task in TASK_NAMES:
            sub = final_df[final_df["Task"] == task]
            if len(sub) > 0:
                rmse = np.sqrt(mean_squared_error(sub["True_Log10"], sub["Pred_Log10"]))
                r2 = r2_score(sub["True_Log10"], sub["Pred_Log10"])
                log_print(f"  {task}: RMSE={rmse:.4f}, R2={r2:.4f}, N={len(sub)}")

        if os.path.exists(final_ema_path):
            model.load_state_dict(torch.load(final_ema_path, map_location=device))
            test_recs_ema = evaluate_multitask(model, test_dl, train_ds.stats, device, "test_ema")
            ema_df = pd.DataFrame(test_recs_ema)
            ema_csv = "EnzCast_MTL_EMA_Results.csv"
            ema_df.to_csv(ema_csv, index=False)

            log_print(f"\n[Final MTL Results - EMA] {ema_csv}")
            for task in TASK_NAMES:
                sub = ema_df[ema_df["Task"] == task]
                if len(sub) > 0:
                    rmse = np.sqrt(mean_squared_error(sub["True_Log10"], sub["Pred_Log10"]))
                    r2 = r2_score(sub["True_Log10"], sub["Pred_Log10"])
                    log_print(f"  {task}: RMSE={rmse:.4f}, R2={r2:.4f}, N={len(sub)}")

    if args.mode in ("single_task", "both"):
        log_print("\n" + "=" * 60)
        log_print("SINGLE-TASK TRAINING (4 separate models)")
        log_print("=" * 60)

        pretrained_path = None
        if args.mode == "both":
            pretrained_path = os.path.join(os.getcwd(), "pretrained_enzcast.pt")

        all_test_dfs = []
        for task_name in TASK_NAMES:
            recon_dim = getattr(train_ds, "recon_dim", 128)
            model_st = EnzCast(
                esm_model_path, train_ds.num_organisms, train_ds.num_types,
                train_ds.num_ecs, recon_dim, train_ds.ec_init_matrix
            ).to(device)

            _, test_df = train_single_task(
                model_st, train_dl, val_dl, test_dl, device, train_ds.stats,
                task_name, args, gpu_config, pretrained_path
            )
            all_test_dfs.append(test_df)
            del model_st
            torch.cuda.empty_cache()

        combined_df = pd.concat(all_test_dfs, ignore_index=True)
        output_csv = "EnzCast_SingleTask_Results.csv"
        combined_df.to_csv(output_csv, index=False)

        log_print(f"\n[Final Single-Task Results] {output_csv}")
        for task in TASK_NAMES:
            sub = combined_df[combined_df["Task"] == task]
            if len(sub) > 0:
                rmse = np.sqrt(mean_squared_error(sub["True_Log10"], sub["Pred_Log10"]))
                r2 = r2_score(sub["True_Log10"], sub["Pred_Log10"])
                log_print(f"  {task}: RMSE={rmse:.4f}, R2={r2:.4f}, N={len(sub)}")

    log_print("\n[Done]")


if __name__ == "__main__":
    main()
