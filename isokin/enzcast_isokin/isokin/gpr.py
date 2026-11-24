from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.special import logsumexp


@dataclass
class GPRNode:
    pass


@dataclass
class GeneNode(GPRNode):
    gene_id: str = ""


@dataclass
class AndNode(GPRNode):
    children: list[GPRNode] = field(default_factory=list)


@dataclass
class OrNode(GPRNode):
    children: list[GPRNode] = field(default_factory=list)


_TOKEN_RE = re.compile(r"\(|\)|\band\b|\bor\b|[^\s()]+", re.IGNORECASE)


def _tokenize(rule_str: str) -> list[str]:
    return _TOKEN_RE.findall(rule_str)


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> GPRNode:
        node = self._expr()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token {self.tokens[self.pos]!r} at position {self.pos}")
        return node

    def _expr(self) -> GPRNode:
        children = [self._term()]
        while self._peek() is not None and self._peek().lower() == "or":
            self._consume()
            children.append(self._term())
        return children[0] if len(children) == 1 else OrNode(children=children)

    def _term(self) -> GPRNode:
        children = [self._factor()]
        while self._peek() is not None and self._peek().lower() == "and":
            self._consume()
            children.append(self._factor())
        return children[0] if len(children) == 1 else AndNode(children=children)

    def _factor(self) -> GPRNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of GPR rule")
        if tok == "(":
            self._consume()
            node = self._expr()
            if self._peek() != ")":
                raise ValueError("Unclosed parenthesis in GPR rule")
            self._consume()
            return node
        if tok == ")":
            raise ValueError("Unexpected closing parenthesis in GPR rule")
        return GeneNode(gene_id=self._consume())


def parse_gpr_rule(rule_str: str | None) -> GPRNode | None:
    if not rule_str or not str(rule_str).strip():
        return None
    tokens = _tokenize(str(rule_str).strip())
    if not tokens:
        return None
    return _Parser(tokens).parse()


def get_all_genes(node: GPRNode | None) -> list[str]:
    if node is None:
        return []
    acc: set[str] = set()
    _collect_genes(node, acc)
    return sorted(acc)


def _collect_genes(node: GPRNode, acc: set[str]) -> None:
    if isinstance(node, GeneNode):
        acc.add(node.gene_id)
    elif isinstance(node, (AndNode, OrNode)):
        for child in node.children:
            _collect_genes(child, acc)


def _expand_to_dnf(node: GPRNode) -> list[list[str]]:
    if isinstance(node, GeneNode):
        return [[node.gene_id]]
    if isinstance(node, OrNode):
        out: list[list[str]] = []
        for child in node.children:
            out.extend(_expand_to_dnf(child))
        return out
    if isinstance(node, AndNode):
        product: list[list[str]] = [[]]
        for child in node.children:
            child_dnf = _expand_to_dnf(child)
            product = [existing + clause for existing in product for clause in child_dnf]
        return product
    return []


def decompose_isoforms(node: GPRNode | None) -> list[GPRNode]:
    if node is None:
        return []
    branches: list[GPRNode] = []
    for clause in _expand_to_dnf(node):
        unique_genes = sorted(set(clause))
        if len(unique_genes) == 1:
            branches.append(GeneNode(gene_id=unique_genes[0]))
        else:
            branches.append(AndNode(children=[GeneNode(gene_id=g) for g in unique_genes]))
    return branches


def decompose_gpr(rule: str, max_isoforms: int | None = None) -> tuple[tuple[str, ...], ...]:
    if not rule or not rule.strip():
        raise ValueError("GPR rule is empty")
    node = parse_gpr_rule(rule)
    raw = _expand_to_dnf(node) if node is not None else []
    normalized = tuple(tuple(dict.fromkeys(clause)) for clause in raw)
    if max_isoforms is not None:
        normalized = normalized[:max_isoforms]
    return normalized


def evaluate_gpr(
    node: GPRNode | None,
    gene_values: dict[str, float],
    default: float = 0.0,
    temperature: float = 1.0,
) -> float:
    if node is None:
        return default
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return _eval_numeric(node, gene_values, default, temperature)


def _eval_numeric(node: GPRNode, gene_values: dict[str, float], default: float, temperature: float) -> float:
    if isinstance(node, GeneNode):
        return float(gene_values.get(node.gene_id, default))
    if isinstance(node, OrNode):
        vals = np.array([_eval_numeric(c, gene_values, default, temperature) for c in node.children], dtype=float)
        return float(temperature * logsumexp(vals / temperature))
    if isinstance(node, AndNode):
        vals = np.array([_eval_numeric(c, gene_values, default, temperature) for c in node.children], dtype=float)
        return float(-temperature * logsumexp(-vals / temperature))
    raise TypeError(f"Unknown GPR node type: {type(node)}")


def evaluate_detection(node: GPRNode | None, cell_gene_detected: dict[str, float]) -> bool:
    if node is None:
        return False
    if isinstance(node, GeneNode):
        return float(cell_gene_detected.get(node.gene_id, 0.0)) > 0.0
    if isinstance(node, OrNode):
        return any(evaluate_detection(c, cell_gene_detected) for c in node.children)
    if isinstance(node, AndNode):
        return all(evaluate_detection(c, cell_gene_detected) for c in node.children)
    raise TypeError(f"Unknown GPR node type: {type(node)}")


def load_gpr_rules(json_path: str | Path) -> dict[str, str]:
    with Path(json_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_isoform_map(
    reaction_ids: list[str],
    gpr_rules: dict[str, str],
    max_isoforms: int = 10,
) -> dict[str, Any]:
    R = len(reaction_ids)
    n_isoforms = np.ones(R, dtype=np.int32)
    isoform_mask = np.zeros((R, max_isoforms), dtype=bool)
    isoform_branches: dict[str, list[GPRNode]] = {}
    isoform_genes: dict[tuple[str, int], list[str]] = {}
    isoform_representative: dict[tuple[str, int], str] = {}
    n_multi_iso = 0
    n_complex_branches = 0

    for r_idx, rxn_id in enumerate(reaction_ids):
        rule_str = gpr_rules.get(rxn_id, "")
        node = parse_gpr_rule(rule_str)
        branches = decompose_isoforms(node)

        if len(branches) == 0:
            n_isoforms[r_idx] = 1
            isoform_mask[r_idx, 0] = True
            isoform_branches[rxn_id] = []
            continue

        n_iso = min(len(branches), max_isoforms)
        n_isoforms[r_idx] = n_iso
        isoform_mask[r_idx, :n_iso] = True
        isoform_branches[rxn_id] = branches[:n_iso]

        if n_iso > 1:
            n_multi_iso += 1

        for i_idx in range(n_iso):
            branch = branches[i_idx]
            genes = get_all_genes(branch)
            if len(genes) > 1:
                n_complex_branches += 1
            isoform_genes[(rxn_id, i_idx)] = genes
            isoform_representative[(rxn_id, i_idx)] = genes[0] if genes else ""

    return {
        "n_isoforms": n_isoforms,
        "isoform_mask": isoform_mask,
        "isoform_branches": isoform_branches,
        "isoform_genes": isoform_genes,
        "isoform_representative": isoform_representative,
        "n_multi_iso": n_multi_iso,
        "n_complex_branches": n_complex_branches,
    }


def build_gene_index(adata) -> dict[str, int]:
    if "GeneSymbol" in adata.var.columns:
        symbols = adata.var["GeneSymbol"].values
    else:
        symbols = adata.var_names.values
        if all(isinstance(s, str) and s.startswith("ENSG") for s in symbols[: min(20, len(symbols))]) and "symbol" in adata.var.columns:
            symbols = adata.var["symbol"].values

    gene_to_idx: dict[str, int] = {}
    for i, sym in enumerate(symbols):
        if isinstance(sym, str) and sym and not sym.startswith("ENSG"):
            gene_to_idx[sym] = i
    return gene_to_idx


def _extract_group_log1p_submatrix(
    adata,
    cell_mask,
    all_var_indices_arr: np.ndarray,
    counts_layer: str | None = "auto",
    target_sum: float = 1e4,
) -> np.ndarray:
    use_counts_layer: str | None = None
    if counts_layer == "auto":
        x_probe = adata[cell_mask[: min(16, len(cell_mask))], :][:, all_var_indices_arr[: min(32, len(all_var_indices_arr))]].X
        if hasattr(x_probe, "toarray"):
            x_probe = x_probe.toarray()
        x_probe = np.asarray(x_probe, dtype=np.float64)
        nz = x_probe[x_probe > 0]
        is_log_normalized = nz.size == 0 or (float(np.max(nz)) <= 30.0 and not np.allclose(nz, np.round(nz)))
        if not is_log_normalized and "counts" in adata.layers:
            use_counts_layer = "counts"
    elif counts_layer not in (None, "X_log1p"):
        use_counts_layer = counts_layer

    if use_counts_layer is not None:
        full_counts = adata[cell_mask, :].layers[use_counts_layer]
        lib_sizes = np.asarray(full_counts.sum(axis=1), dtype=np.float64).reshape(-1, 1)
        sub_counts = full_counts[:, all_var_indices_arr]
        if hasattr(sub_counts, "toarray"):
            sub_counts = sub_counts.toarray()
        sub_counts = np.asarray(sub_counts, dtype=np.float64)
        scale = np.where(lib_sizes > 0.0, target_sum / lib_sizes, 0.0)
        return np.log1p(sub_counts * scale)

    x_sub = adata[cell_mask, :][:, all_var_indices_arr].X
    if hasattr(x_sub, "toarray"):
        x_sub = x_sub.toarray()
    x_sub = np.asarray(x_sub, dtype=np.float64)
    if counts_layer == "auto" and x_sub.size > 0 and float(np.max(x_sub)) > 30.0:
        full_x = adata[cell_mask, :].X
        lib_sizes = np.asarray(full_x.sum(axis=1), dtype=np.float64).reshape(-1, 1)
        scale = np.where(lib_sizes > 0.0, target_sum / lib_sizes, 0.0)
        return np.log1p(x_sub * scale)
    return x_sub


def compute_isoform_expression(
    adata,
    reaction_ids: list[str],
    isoform_map: dict[str, Any],
    patient_col: str = "patient_id",
    celltype_col: str = "cell_type",
    counts_layer: str | None = "auto",
    kappa: float = 1.0,
    pseudocount: float = 1e-6,
    temperature: float = 1.0,
    max_isoforms: int = 10,
    target_sum_cp10k: float = 1e4,
) -> dict[str, Any]:
    gene_to_idx = build_gene_index(adata)
    gene_to_idx_upper: dict[str, int] = {}
    for sym, idx in gene_to_idx.items():
        gene_to_idx_upper.setdefault(sym.upper(), idx)

    def _resolve_gene_var_idx(gene_symbol: str) -> int | None:
        if gene_symbol in gene_to_idx:
            return gene_to_idx[gene_symbol]
        return gene_to_idx_upper.get(gene_symbol.upper())

    R = len(reaction_ids)
    rxn_to_col = {r: i for i, r in enumerate(reaction_ids)}
    isoform_branches = isoform_map["isoform_branches"]
    isoform_mask = isoform_map["isoform_mask"]

    all_needed_genes: set[str] = set()
    for rxn_id in reaction_ids:
        for branch in isoform_branches.get(rxn_id, []):
            all_needed_genes.update(get_all_genes(branch))

    resolved_needed = {g: vidx for g in all_needed_genes if (vidx := _resolve_gene_var_idx(g)) is not None}
    all_var_indices = sorted(set(resolved_needed.values()))
    var_idx_to_local = {v: i for i, v in enumerate(all_var_indices)}
    all_var_indices_arr = np.array(all_var_indices, dtype=int) if all_var_indices else np.array([], dtype=int)

    iso_gene_var: dict[tuple[str, int], dict[str, int]] = {}
    reactions_with_observed_gene: set[str] = set()
    for rxn_id in reaction_ids:
        for i_idx, branch in enumerate(isoform_branches.get(rxn_id, [])):
            mapping = {g: var_idx_to_local[resolved_needed[g]] for g in get_all_genes(branch) if g in resolved_needed}
            if mapping:
                reactions_with_observed_gene.add(rxn_id)
            iso_gene_var[(rxn_id, i_idx)] = mapping

    groups = adata.obs.groupby([patient_col, celltype_col], observed=True)
    group_keys = sorted(groups.groups.keys())
    G = len(group_keys)

    log10_E_iso = np.full((G, R, max_isoforms), np.log10(pseudocount), dtype=np.float64)
    detection_rate_iso = np.zeros((G, R, max_isoforms), dtype=np.float64)
    group_n_cells = np.zeros(G, dtype=np.int64)

    for g_idx, (patient, celltype) in enumerate(group_keys):
        cell_mask = groups.groups[(patient, celltype)]
        n_cells = len(cell_mask)
        group_n_cells[g_idx] = n_cells
        if n_cells == 0 or len(all_var_indices_arr) == 0:
            continue

        x_sub = _extract_group_log1p_submatrix(
            adata=adata,
            cell_mask=cell_mask,
            all_var_indices_arr=all_var_indices_arr,
            counts_layer=counts_layer,
            target_sum=target_sum_cp10k,
        )
        x_linear = np.expm1(x_sub)
        mean_linear = x_linear.mean(axis=0)
        detected_per_cell = x_sub > 0.0

        for rxn_id in reaction_ids:
            r_idx = rxn_to_col[rxn_id]
            for i_idx, branch in enumerate(isoform_branches.get(rxn_id, [])):
                gene_local = iso_gene_var.get((rxn_id, i_idx), {})
                if not gene_local:
                    continue
                gene_values = {g: float(mean_linear[loc]) for g, loc in gene_local.items()}
                e_raw = evaluate_gpr(branch, gene_values, default=0.0, temperature=temperature)
                e_scaled = kappa * max(e_raw, 0.0) + pseudocount
                log10_E_iso[g_idx, r_idx, i_idx] = np.log10(e_scaled)

                if isinstance(branch, GeneNode):
                    loc = next(iter(gene_local.values()))
                    detection_rate_iso[g_idx, r_idx, i_idx] = float(detected_per_cell[:, loc].mean())
                elif isinstance(branch, AndNode):
                    all_genes = get_all_genes(branch)
                    if len(gene_local) < len(all_genes):
                        detection_rate_iso[g_idx, r_idx, i_idx] = 0.0
                    else:
                        locs = [gene_local[g] for g in all_genes]
                        detection_rate_iso[g_idx, r_idx, i_idx] = float(detected_per_cell[:, locs].all(axis=1).mean())
                else:
                    n_det = sum(
                        1
                        for ci in range(n_cells)
                        if evaluate_detection(branch, {g: float(detected_per_cell[ci, loc]) for g, loc in gene_local.items()})
                    )
                    detection_rate_iso[g_idx, r_idx, i_idx] = n_det / n_cells

    e_linear = np.where(isoform_mask[np.newaxis, :, :], np.power(10.0, log10_E_iso), 0.0)
    log10_E_total = np.log10(np.maximum(e_linear.sum(axis=-1), pseudocount))

    return {
        "log10_E_iso": log10_E_iso,
        "detection_rate_iso": detection_rate_iso,
        "log10_E_total": log10_E_total,
        "group_labels": [(str(p), str(c)) for p, c in group_keys],
        "group_n_cells": group_n_cells,
        "reaction_ids": reaction_ids,
        "reaction_to_idx": rxn_to_col,
        "reactions_with_observed_gene": reactions_with_observed_gene,
    }
