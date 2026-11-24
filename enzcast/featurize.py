#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

RADIUS_CUTOFF = 10.0
KNN_K = 20

AA_MAP = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "SEC": 4,
    "PYL": 11,
}

ATOM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "H",
    "Na",
    "K",
    "Ca",
    "Mg",
    "Fe",
    "Zn",
    "Mn",
    "Cu",
    "Co",
]


def _one_hot_encoding(val, allowed_set):
    encoding = [0] * (len(allowed_set) + 1)
    if val in allowed_set:
        encoding[allowed_set.index(val)] = 1
    else:
        encoding[-1] = 1
    return encoding


def morgan_fingerprint(smiles: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return torch.zeros(2048, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr).to(torch.float32)


def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        hybridization = atom.GetHybridization()
        feat = (
            _one_hot_encoding(symbol, ATOM_LIST)
            + _one_hot_encoding(
                str(atom.GetChiralTag()),
                ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
            )
            + _one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6])
            + _one_hot_encoding(
                hybridization,
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )
            + [atom.GetFormalCharge()]
            + [atom.GetTotalNumHs()]
            + [1 if atom.GetIsAromatic() else 0]
        )
        atom_feats.append(feat)

    x = torch.tensor(atom_feats, dtype=torch.float32)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        attr = [
            1 if btype == Chem.rdchem.BondType.SINGLE else 0,
            1 if btype == Chem.rdchem.BondType.DOUBLE else 0,
            1 if btype == Chem.rdchem.BondType.TRIPLE else 0,
            1 if btype == Chem.rdchem.BondType.AROMATIC else 0,
            1 if bond.GetIsConjugated() else 0,
            1 if bond.IsInRing() else 0,
        ]
        edge_indices.append([i, j])
        edge_attrs.append(attr)
        edge_indices.append([j, i])
        edge_attrs.append(attr)

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def parse_pdb_to_tensor(pdb_path: str | Path):
    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        return None, None, None

    coords = []
    aa_indices = []
    res_ids = []
    target_chain = None

    with pdb_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain_id = line[21]
                if target_chain is None:
                    target_chain = chain_id
                if chain_id != target_chain:
                    continue
                res_name = line[17:20].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_seq = int(line[22:26])
                    coords.append([x, y, z])
                    aa_indices.append(AA_MAP.get(res_name, 20))
                    res_ids.append(res_seq)
                except ValueError:
                    continue

    if not coords:
        return None, None, None

    order = np.argsort(res_ids)
    coords_t = torch.tensor(coords, dtype=torch.float32)[order]
    aa_t = torch.tensor(aa_indices, dtype=torch.long)[order]
    res_t = torch.tensor(res_ids, dtype=torch.long)[order]
    return coords_t, aa_t, res_t


def _pairwise_radius_and_knn(pos: torch.Tensor, radius: float = RADIUS_CUTOFF, k: int = KNN_K):
    try:
        from torch_cluster import knn_graph, radius_graph

        r_edges = radius_graph(pos, r=radius, loop=False, max_num_neighbors=64)
        k_edges = knn_graph(pos, k=min(k, max(1, pos.size(0) - 1)), loop=False)
        return r_edges, k_edges
    except ImportError:
        n = pos.size(0)
        if n <= 1:
            empty = torch.empty((2, 0), dtype=torch.long)
            return empty, empty
        dist = torch.cdist(pos, pos)
        eye = torch.eye(n, dtype=torch.bool)
        r_mask = (dist <= radius) & (~eye)
        r_src, r_dst = torch.where(r_mask)
        r_edges = torch.stack([r_src, r_dst], dim=0)

        dist_knn = dist.masked_fill(eye, float("inf"))
        k_eff = min(k, n - 1)
        knn_idx = torch.topk(dist_knn, k=k_eff, dim=1, largest=False).indices
        dst = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        src = knn_idx.reshape(-1)
        k_edges = torch.stack([src, dst], dim=0)
        return r_edges, k_edges


def construct_gearnet_graph(pos: torch.Tensor, x: torch.Tensor, res_ids: torch.Tensor) -> Data:
    num_nodes = pos.size(0)
    edge_index_list = []
    edge_type_list = []

    def add_edges(edges: torch.Tensor, type_idx: int):
        if edges.size(1) > 0:
            edge_index_list.append(edges)
            edge_type_list.append(torch.full((edges.size(1),), type_idx, dtype=torch.long))

    if num_nodes > 1:
        src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        consec = (res_ids[dst] - res_ids[src]) == 1
        v_src = src[consec]
        v_dst = dst[consec]
        if v_src.numel() > 0:
            seq_edges = torch.stack([torch.cat([v_src, v_dst]), torch.cat([v_dst, v_src])], dim=0)
            add_edges(seq_edges, 0)

        r_edges, k_edges = _pairwise_radius_and_knn(pos)
        add_edges(r_edges, 1)
        add_edges(k_edges, 2)

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return Data(x=x, pos=pos, edge_index=edge_index, edge_type=edge_type)


def pdb_to_protein_graph(pdb_path: str | Path) -> Data | None:
    pos, x, res_ids = parse_pdb_to_tensor(pdb_path)
    if pos is None:
        return None
    return construct_gearnet_graph(pos, x, res_ids)


def build_substrate_cache_cli(input_csvs: list[Path], output_dir: Path, smiles_col: str = "SMILES") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_smiles: set[str] = set()
    for csv_path in input_csvs:
        df = pd.read_csv(csv_path, low_memory=False)
        col = smiles_col if smiles_col in df.columns else ("Smiles" if "Smiles" in df.columns else None)
        if col is None:
            raise ValueError(f"Column {smiles_col!r} not found in {csv_path}")
        unique_smiles.update(str(s).strip() for s in df[col].dropna().unique() if str(s).strip())

    written = 0
    for smi in tqdm(sorted(unique_smiles), desc="Building substrate graphs"):
        digest = hashlib.md5(smi.encode("utf-8")).hexdigest()
        out_pt = output_dir / f"{digest}.pt"
        if out_pt.exists():
            written += 1
            continue
        graph = smiles_to_graph(smi)
        if graph is not None:
            torch.save(graph, out_pt)
            written += 1
    return written


def build_protein_cache_cli(pdb_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    written = 0
    for pdb_path in tqdm(pdb_files, desc="Building protein graphs"):
        uid = pdb_path.stem
        out_pt = output_dir / f"{uid}.pt"
        if out_pt.exists():
            written += 1
            continue
        graph = pdb_to_protein_graph(pdb_path)
        if graph is not None:
            torch.save(graph, out_pt)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute EnzCast protein or substrate graph caches")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sub = sub.add_parser("substrates", help="Build substrate molecular graph cache from CSV(s)")
    p_sub.add_argument("--input-csv", nargs="+", type=Path, required=True)
    p_sub.add_argument("--output-dir", type=Path, required=True)
    p_sub.add_argument("--smiles-col", default="SMILES")

    p_prot = sub.add_parser("proteins", help="Build C-alpha GearNet graph cache from PDB directory")
    p_prot.add_argument("--pdb-dir", type=Path, required=True)
    p_prot.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "substrates":
        count = build_substrate_cache_cli(args.input_csv, args.output_dir, args.smiles_col)
        print(f"Substrate graphs ready: {count} files in {args.output_dir}")
    else:
        count = build_protein_cache_cli(args.pdb_dir, args.output_dir)
        print(f"Protein graphs ready: {count} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
