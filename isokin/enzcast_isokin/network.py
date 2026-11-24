from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

_COMPARTMENTS = ("c", "m", "e", "x", "r", "n", "l")


def _expand_compartments(base_ids: list[str], compartments: tuple[str, ...] | list[str] | None = None) -> frozenset[str]:
    comps = _COMPARTMENTS if compartments is None else compartments
    return frozenset(f"{b}_{c}" for b in base_ids for c in comps)


EXCLUDED_COFACTORS: frozenset[str] = frozenset().union(
    _expand_compartments(["h"]),
    _expand_compartments(["h2o"]),
    _expand_compartments(["atp", "adp", "amp"], ["c", "m"]),
    _expand_compartments(["gtp", "gdp"], ["c", "m"]),
    _expand_compartments(["nad", "nadh"], ["c", "m"]),
    _expand_compartments(["nadp", "nadph"], ["c", "m"]),
    _expand_compartments(["coa"], ["c", "m"]),
    _expand_compartments(["fad", "fadh2"], ["c", "m"]),
    _expand_compartments(["pi", "ppi"], ["c", "m", "e"]),
    _expand_compartments(["co2", "o2", "nh4"], ["c", "m", "e"]),
)


def load_recon3d_stoichiometry(json_path: str | Path) -> dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as fh:
        model = json.load(fh)

    reactions: dict[str, dict[str, Any]] = {}
    for rxn in model.get("reactions", []):
        rxn_id = rxn["id"]
        reactions[rxn_id] = {
            "metabolites": rxn.get("metabolites", {}),
            "subsystem": rxn.get("subsystem", ""),
            "gene_reaction_rule": rxn.get("gene_reaction_rule", ""),
            "lower_bound": rxn.get("lower_bound", -1000),
            "upper_bound": rxn.get("upper_bound", 1000),
        }

    metabolites: dict[str, dict[str, Any]] = {}
    for met in model.get("metabolites", []):
        metabolites[met["id"]] = {
            "name": met.get("name", ""),
            "compartment": met.get("compartment", ""),
        }
    return {"reactions": reactions, "metabolites": metabolites}


def build_stoichiometric_matrix(
    recon3d_data: dict[str, Any],
    reaction_ids: list[str],
    excluded_metabolites: frozenset[str] = EXCLUDED_COFACTORS,
    min_reactions_per_metabolite: int = 2,
) -> tuple[np.ndarray, list[str], list[str], dict[str, dict[str, Any]]]:
    all_rxns = recon3d_data["reactions"]
    met_rxn_count: dict[str, int] = defaultdict(int)
    met_producing: dict[str, set[str]] = defaultdict(set)
    met_consuming: dict[str, set[str]] = defaultdict(set)

    found_rxn_ids: list[str] = []
    for rxn_id in reaction_ids:
        if rxn_id not in all_rxns:
            continue
        found_rxn_ids.append(rxn_id)
        for met_id, coeff in all_rxns[rxn_id].get("metabolites", {}).items():
            if met_id in excluded_metabolites:
                continue
            met_rxn_count[met_id] += 1
            if coeff > 0:
                met_producing[met_id].add(rxn_id)
            elif coeff < 0:
                met_consuming[met_id].add(rxn_id)

    met_info: dict[str, dict[str, Any]] = {}
    internal_mets: list[str] = []
    for met_id, n_rxns in met_rxn_count.items():
        if met_id in excluded_metabolites:
            met_info[met_id] = {"type": "excluded", "n_reactions": n_rxns}
            continue
        has_prod = len(met_producing.get(met_id, set())) > 0
        has_cons = len(met_consuming.get(met_id, set())) > 0
        if has_prod and has_cons and n_rxns >= min_reactions_per_metabolite:
            met_info[met_id] = {"type": "internal", "n_reactions": n_rxns}
            internal_mets.append(met_id)
        else:
            met_info[met_id] = {"type": "boundary", "n_reactions": n_rxns}

    internal_mets = sorted(internal_mets)

    rxn_in_internal: set[str] = set()
    for met_id in internal_mets:
        rxn_in_internal |= met_producing.get(met_id, set())
        rxn_in_internal |= met_consuming.get(met_id, set())
    filtered_rxn_ids = [r for r in found_rxn_ids if r in rxn_in_internal]

    met_to_idx = {m: i for i, m in enumerate(internal_mets)}
    rxn_to_idx = {r: i for i, r in enumerate(filtered_rxn_ids)}

    S = np.zeros((len(internal_mets), len(filtered_rxn_ids)), dtype=np.float64)
    for rxn_id in filtered_rxn_ids:
        r_idx = rxn_to_idx[rxn_id]
        for met_id, coeff in all_rxns[rxn_id].get("metabolites", {}).items():
            if met_id in met_to_idx:
                S[met_to_idx[met_id], r_idx] = coeff

    return S, internal_mets, filtered_rxn_ids, met_info


def compute_metabolite_degrees(S: np.ndarray) -> np.ndarray:
    return np.count_nonzero(S, axis=1).astype(np.int64)


def find_connected_components(
    S: np.ndarray,
    met_ids: list[str],
    rxn_ids: list[str],
    min_metabolites: int = 3,
) -> list[dict[str, Any]]:
    M, R = S.shape
    if R == 0:
        return []

    parent = list(range(R))
    uf_rank = [0] * R

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if uf_rank[ra] < uf_rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if uf_rank[ra] == uf_rank[rb]:
            uf_rank[ra] += 1

    for m in range(M):
        nz = np.nonzero(S[m, :])[0]
        for j in range(1, len(nz)):
            union(int(nz[0]), int(nz[j]))

    comp_rxns: dict[int, list[int]] = defaultdict(list)
    for r in range(R):
        comp_rxns[find(r)].append(r)

    components: list[dict[str, Any]] = []
    for rxn_indices in comp_rxns.values():
        met_indices = sorted({m for m in range(M) if any(S[m, r] != 0 for r in rxn_indices)})
        if len(met_indices) >= min_metabolites:
            sorted_rxns = sorted(rxn_indices)
            components.append(
                {
                    "metabolite_indices": met_indices,
                    "reaction_indices": sorted_rxns,
                    "metabolite_ids": [met_ids[i] for i in met_indices],
                    "reaction_ids": [rxn_ids[j] for j in sorted_rxns],
                    "n_metabolites": len(met_indices),
                    "n_reactions": len(sorted_rxns),
                }
            )

    components.sort(key=lambda c: c["n_reactions"], reverse=True)
    return components


def build_network(
    recon3d_json_path: str | Path,
    common_reaction_ids: list[str],
    excluded_cofactors: frozenset[str] = EXCLUDED_COFACTORS,
    min_reactions_per_metabolite: int = 2,
    min_metabolites_per_component: int = 3,
    recon3d_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recon3d = recon3d_cache if recon3d_cache is not None else load_recon3d_stoichiometry(recon3d_json_path)
    S, met_ids, rxn_ids, met_info = build_stoichiometric_matrix(
        recon3d,
        common_reaction_ids,
        excluded_cofactors,
        min_reactions_per_metabolite,
    )
    met_degrees = compute_metabolite_degrees(S)
    components = find_connected_components(S, met_ids, rxn_ids, min_metabolites_per_component)

    constrained: set[str] = set()
    for comp in components:
        constrained.update(comp["reaction_ids"])
    unconstrained = [r for r in common_reaction_ids if r not in constrained]

    return {
        "S": S,
        "metabolite_ids": met_ids,
        "reaction_ids": rxn_ids,
        "reaction_to_idx": {r: i for i, r in enumerate(rxn_ids)},
        "metabolite_to_idx": {m: i for i, m in enumerate(met_ids)},
        "metabolite_info": met_info,
        "components": components,
        "metabolite_degrees": met_degrees,
        "n_metabolites": len(met_ids),
        "n_reactions": len(rxn_ids),
        "n_unconstrained_reactions": len(unconstrained),
        "unconstrained_reaction_ids": unconstrained,
    }
