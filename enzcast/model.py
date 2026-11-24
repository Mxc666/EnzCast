from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from transformers import AutoConfig, AutoModel

SEQ_MAX_LEN = 1024
SUBSTRATE_IN_DIM = 42
HIDDEN_DIM = 256
COND_DIM = 128
DROPOUT = 0.15

EMA_DECAY = 0.999
BETA_NLL_BETA = 0.5
ESM_EXTRACT_LAYERS = [10, 21, 32]


def detect_gpu_config(device_id: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {
            "gpu_type": "cpu",
            "precision": "float32",
            "use_scaler": False,
            "batch_size": 2,
            "num_workers": 0,
            "tf32": False,
        }

    gpu_name = torch.cuda.get_device_name(device_id).lower()
    if "a800" in gpu_name or "a100" in gpu_name or "h100" in gpu_name or "h800" in gpu_name:
        return {
            "gpu_type": "Ampere80G",
            "precision": "bfloat16",
            "use_scaler": False,
            "batch_size": 22,
            "num_workers": 8,
            "tf32": True,
        }
    if any(x in gpu_name for x in ["rtx 30", "rtx 40", "a6000", "a5000", "a4000", "l40"]):
        return {
            "gpu_type": "Ampere",
            "precision": "bfloat16",
            "use_scaler": False,
            "batch_size": 6,
            "num_workers": 4,
            "tf32": True,
        }
    if "v100" in gpu_name:
        return {
            "gpu_type": "V100",
            "precision": "float16",
            "use_scaler": True,
            "batch_size": 4,
            "num_workers": 4,
            "tf32": False,
        }
    return {
        "gpu_type": "GenericCUDA",
        "precision": "float16",
        "use_scaler": True,
        "batch_size": 4,
        "num_workers": 4,
        "tf32": False,
    }


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class ProteinLLMAdapter(nn.Module):
    def __init__(
        self,
        model_path: str,
        dim: int = HIDDEN_DIM,
        extract_layers: list[int] | None = None,
        from_config_only: bool = False,
    ):
        super().__init__()
        self.extract_layers = extract_layers or ESM_EXTRACT_LAYERS

        if from_config_only:
            esm_cfg = AutoConfig.from_pretrained(model_path)
            esm_cfg.output_hidden_states = True
            base_esm = AutoModel.from_config(esm_cfg)
        else:
            base_esm = AutoModel.from_pretrained(model_path, output_hidden_states=True)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
        )
        self.esm = get_peft_model(base_esm, lora_cfg)

        esm_hidden_size = self.esm.config.hidden_size
        n_layers_extracted = len(self.extract_layers)

        self.layer_weights = nn.Parameter(torch.ones(n_layers_extracted) / n_layers_extracted)
        self.layer_norm_per = nn.ModuleList(
            [nn.LayerNorm(esm_hidden_size) for _ in range(n_layers_extracted)]
        )

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

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        if ids.shape[1] > SEQ_MAX_LEN:
            ids = ids[:, :SEQ_MAX_LEN]
            mask = mask[:, :SEQ_MAX_LEN]

        outputs = self.esm(ids, mask)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            weights = F.softmax(self.layer_weights, dim=0)
            combined = torch.zeros_like(hidden_states[-1])
            for i, layer_idx in enumerate(self.extract_layers):
                if layer_idx < len(hidden_states):
                    combined = combined + weights[i] * self.layer_norm_per[i](hidden_states[layer_idx])
        else:
            combined = outputs.last_hidden_state

        h = self.proj(combined)
        h_global = self.attn_pool(h, mask)
        return self.norm(h), self.norm(h_global)


class MultiScaleEGCL(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1 + edge_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, x, edge_index, edge_attr):
        row, col = edge_index
        coord_diff = x[row] - x[col]
        radial = (coord_diff**2).sum(dim=-1, keepdim=True)

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
    def __init__(self, dim: int, n_heads: int = 4):
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
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
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

        attn_list = [pyg_softmax(scores[:, h_idx], batch) for h_idx in range(self.n_heads)]
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
    def __init__(self, dim: int = HIDDEN_DIM, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.node_embed = nn.Embedding(21, dim)
        self.edge_embed = nn.Embedding(3, dim)
        self.layers = nn.ModuleList([HierarchicalVNLayer(dim, n_heads) for _ in range(n_layers)])
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, data):
        h = torch.clamp(data.x.long(), 0, 20)
        h = self.node_embed(h)

        x = torch.nan_to_num(data.pos.float(), nan=0.0, posinf=0.0, neginf=0.0)
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
    def __init__(self, in_dim: int = SUBSTRATE_IN_DIM, hidden_dim: int = HIDDEN_DIM, n_layers: int = 3):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
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
                pooled_features.append(self.multi_pool_proj(multi_pool))
            else:
                pooled_features.append(torch.zeros(1, x.size(-1), device=x.device))

        h_sub = torch.cat(pooled_features, dim=0)
        return self.out_norm(h_sub), x, batch


class PocketAwareCrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
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

            q = self.q_proj(h_sub_global[b : b + 1]).unsqueeze(0)
            k = self.k_proj(h_p).unsqueeze(0)
            v = self.v_proj(h_p).unsqueeze(0)

            q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(1, -1)
            outputs.append(out)

        h_cross = torch.cat(outputs, dim=0)
        return self.norm(self.out_proj(h_cross))


class EnhancedConditionEncoder(nn.Module):
    def __init__(
        self,
        num_species: int,
        num_types: int,
        num_ecs: int,
        recon_dim: int,
        ec_init_matrix: torch.Tensor | None = None,
        embed_dim: int = COND_DIM,
    ):
        super().__init__()
        self.species_emb = nn.Embedding(num_species, embed_dim)
        self.type_emb = nn.Embedding(num_types, embed_dim)

        self.ph_enc = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.temp_enc = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
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
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, c: dict[str, torch.Tensor]):
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
    def __init__(self, feat_dim: int, cond_dim: int, n_experts: int = 4):
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

    def forward(self, x_feat: torch.Tensor, z_cond: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared_trunk(torch.cat([x_feat, z_cond], dim=-1))
        h_binding = self.binding_trunk(h)
        out_km = self.km_head(h_binding)
        out_ki = self.ki_head(h_binding)

        h_catalytic = self.catalytic_trunk(h)
        out_kcat = self.kcat_head(h_catalytic)
        out_eff = self.eff_head(h_catalytic)
        return {"km": out_km, "kcat": out_kcat, "eff": out_eff, "ki": out_ki}


class EnzCastModel(nn.Module):
    def __init__(
        self,
        esm_path: str,
        num_species: int,
        num_types: int,
        num_ecs: int,
        recon_dim: int,
        ec_init_matrix: torch.Tensor | None = None,
        from_config_only: bool = False,
    ):
        super().__init__()
        self.seq_enc = ProteinLLMAdapter(
            esm_path,
            dim=HIDDEN_DIM,
            from_config_only=from_config_only,
        )
        self.struct_enc = AdvancedStructureEncoder(dim=HIDDEN_DIM, n_layers=4, n_heads=4)
        self.sub_enc = SubstrateTransformerMultiPool(
            in_dim=SUBSTRATE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=3,
        )
        self.cond_enc = EnhancedConditionEncoder(
            num_species,
            num_types,
            num_ecs,
            recon_dim,
            ec_init_matrix,
            embed_dim=COND_DIM,
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
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 128),
        )
        self.sub_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 128),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_representations(self, ids, mask, struct, sub_graph, autocast_dtype=None):
        h_seq_res, h_seq_global = self.seq_enc(ids, mask)

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if struct.x.is_cuda
            else nullcontext()
        )
        with amp_ctx:
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

    def contrastive_forward(self, ids, mask, struct, sub_graph, autocast_dtype=None):
        reps = self.get_representations(ids, mask, struct, sub_graph, autocast_dtype)
        h_prot = torch.cat([reps["h_seq_global"], reps["h_struct_vn"]], dim=-1)
        z_prot = F.normalize(self.prot_proj(h_prot), dim=-1, eps=1e-6)
        z_sub = F.normalize(self.sub_proj(reps["h_sub_global"]), dim=-1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * z_prot @ z_sub.t()

    def forward(self, ids, mask, struct, sub_graph, phys, autocast_dtype=torch.bfloat16):
        reps = self.get_representations(ids, mask, struct, sub_graph, autocast_dtype)
        B = reps["B"]

        z_cond, scales, _ = self.cond_enc(phys)
        z_cond = z_cond[:B]
        scales = scales[:B]

        h_cross = self.cross_attn(
            reps["h_sub_atoms"],
            reps["h_struct_res"],
            reps["struct_batch"],
            reps["sub_batch"],
        )[:B]

        h_combined = torch.cat(
            [
                reps["h_seq_global"] * scales[:, 0:1],
                reps["h_struct_vn"] * scales[:, 1:2],
                reps["h_sub_global"] * scales[:, 2:3],
                h_cross,
            ],
            dim=-1,
        )

        h_fused = self.fusion(h_combined)
        gate = self.fusion_gate(h_combined)
        h_final = h_fused * gate
        return self.prediction_head(h_final, z_cond)


class BetaNLLLoss(nn.Module):
    def __init__(self, beta: float = BETA_NLL_BETA):
        super().__init__()
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, nll_weight: float = 1.0):
        mu = mu.view(-1)
        logvar = logvar.view(-1).clamp(-8.0, 8.0)
        target = target.view(-1)

        var = torch.exp(logvar)
        diff = target - mu
        nll = 0.5 * logvar + 0.5 * (diff**2) / var.clamp(min=1e-6)
        if self.beta > 0:
            nll = nll * (var.detach() ** self.beta)
        return nll.mean() * nll_weight


class UncertaintyWeightedMTL(nn.Module):
    def __init__(self, n_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses_dict: dict[str, torch.Tensor | None], task_names: list[str] | None = None):
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

    def get_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.exp(-torch.clamp(self.log_vars, -6.0, 6.0))


def consistency_loss_log10(mu_kcat: torch.Tensor, mu_km: torch.Tensor, mu_eff: torch.Tensor, stats: dict) -> torch.Tensor:
    s0 = stats[0] if 0 in stats else stats["0"]
    s1 = stats[1] if 1 in stats else stats["1"]
    s2 = stats[2] if 2 in stats else stats["2"]

    raw_km = mu_km * s0["std"] + s0["mean"]
    raw_kcat = mu_kcat * s1["std"] + s1["mean"]
    raw_eff = mu_eff * s2["std"] + s2["mean"]
    return ((raw_kcat - raw_km - raw_eff) ** 2).mean()


def get_nll_weight_short(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return 0.5
    nll_floor = 0.05
    ramp_end_epoch = total_epochs * 0.80
    if epoch >= ramp_end_epoch:
        return 1.0
    progress = epoch / ramp_end_epoch
    return nll_floor + (1.0 - nll_floor) * progress
