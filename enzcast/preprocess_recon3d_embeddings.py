#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def build_reaction_metabolite_graph(model) -> nx.Graph:
    graph = nx.Graph()
    for rxn in model.reactions:
        graph.add_node(rxn.id, type="reaction")
        for met in rxn.metabolites:
            if met.id not in graph:
                graph.add_node(met.id, type="metabolite")
            graph.add_edge(rxn.id, met.id)

    metabolite_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "metabolite"]
    degrees = dict(graph.degree(metabolite_nodes))
    sorted_mets = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    num_to_remove = int(len(metabolite_nodes) * 0.02)
    explicit_currency = {"h", "h2o", "atp", "adp", "pi", "ppi", "nad", "nadh", "coa", "o2", "co2"}

    nodes_to_remove = {sorted_mets[i][0] for i in range(num_to_remove)}
    for met_id in metabolite_nodes:
        if met_id.split("_")[0].lower() in explicit_currency:
            nodes_to_remove.add(met_id)

    graph.remove_nodes_from(nodes_to_remove)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def generate_recon3d_embeddings(
    recon3d_json: Path,
    output_csv: Path,
    embedding_dim: int = 128,
    walk_length: int = 30,
    num_walks: int = 200,
    window_size: int = 10,
    workers: int = 4,
) -> None:
    import cobra
    from node2vec import Node2Vec

    model = cobra.io.load_json_model(str(recon3d_json))
    graph = build_reaction_metabolite_graph(model)
    n2v = Node2Vec(
        graph,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=1,
        q=1,
        workers=workers,
        quiet=True,
    )
    fitted = n2v.fit(window=window_size, min_count=1, batch_words=4)

    vectors_in_model = [fitted.wv[n] for n in graph.nodes() if n in fitted.wv]
    global_mean = np.mean(vectors_in_model, axis=0) if vectors_in_model else np.zeros(embedding_dim)

    rows = []
    for rxn in model.reactions:
        rid = rxn.id
        vec = fitted.wv[rid] if rid in fitted.wv else global_mean
        ec = rxn.annotation.get("ec-code", np.nan)
        if isinstance(ec, list):
            ec = ec[0]
        row = {"Reaction_ID": rid, "EC_Number_Recon": str(ec) if ec else np.nan}
        for i, val in enumerate(vec):
            row[f"ReacFeat_{i}"] = float(val)
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Recon3D reaction Node2Vec embeddings")
    parser.add_argument("--recon3d-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    generate_recon3d_embeddings(args.recon3d_json, args.output_csv, args.embedding_dim, workers=args.workers)
    print(f"Saved Recon3D reaction embeddings to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
