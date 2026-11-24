from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pandas as pd

from enzcast_isokin.config import PipelineConfig
from enzcast_isokin.data import validate_anndata
from enzcast_isokin.enzcast import PriorTable
from enzcast_isokin.pipeline import KineticsPipeline


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enzcast-isokin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-h5ad", help="Validate an input h5ad transcriptomics file")
    _add_h5ad_args(validate_parser)
    validate_parser.set_defaults(func=_validate_h5ad)

    priors_parser = subparsers.add_parser("make-priors", help="Build a long-form EnzCast prior table from registered CSVs")
    priors_parser.add_argument("--project-root", default=".", help="Project root containing Dataset/isokin")
    priors_parser.add_argument("--out", required=True, help="Output long-form prior CSV path")
    priors_parser.set_defaults(func=_make_priors)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Fit flux-constrained, GPR-resolved IsoKin posteriors on an h5ad cohort",
    )
    _add_h5ad_args(infer_parser)
    infer_parser.add_argument("--project-root", default=".", help="Project root containing Dataset/isokin")
    infer_parser.add_argument("--out", required=True, help="Output directory for posterior CSVs")
    infer_parser.add_argument("--max-iterations", type=int, default=None, help="Override SVI max iterations")
    infer_parser.add_argument("--posterior-samples", type=int, default=None, help="Override posterior sample count")
    infer_parser.add_argument("--device", default=None, help="Device override ('auto', 'cpu', or 'cuda')")
    infer_parser.add_argument(
        "--observed-genes-only",
        action="store_true",
        help="Restrict reactions to those with at least one GPR gene present in AnnData",
    )
    infer_parser.add_argument(
        "--min-component-metabolites",
        type=int,
        default=None,
        help="Override minimum internal metabolites per connected network component",
    )
    infer_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    infer_parser.set_defaults(func=_infer)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize an IsoKin output directory")
    summarize_parser.add_argument("--input", required=True, help="Output directory produced by enzcast-isokin infer")
    summarize_parser.set_defaults(func=_summarize)
    return parser


def _add_h5ad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input h5ad path")
    parser.add_argument("--patient-col", required=True, help="AnnData obs column for patient/sample ID")
    parser.add_argument("--celltype-col", required=True, help="AnnData obs column for cell type or state")
    parser.add_argument(
        "--counts-layer",
        default="auto",
        help="Expression source: 'auto' (uses log1p X or normalizes layers['counts']), layer name, or 'X_log1p'",
    )


def _validate_h5ad(args) -> int:
    import anndata as ad

    adata = ad.read_h5ad(Path(args.input).expanduser().resolve())
    layer_arg = None if args.counts_layer == "X_log1p" else args.counts_layer
    report = validate_anndata(
        adata,
        patient_col=args.patient_col,
        celltype_col=args.celltype_col,
        counts_layer=layer_arg,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def _make_priors(args) -> int:
    config = PipelineConfig.default(project_root=args.project_root)
    priors = PriorTable.from_artifact_registry(config.artifacts)
    priors.to_csv(Path(args.out).expanduser().resolve())
    print(json.dumps(priors.summary(), indent=2, sort_keys=True))
    return 0


def _infer(args) -> int:
    config = PipelineConfig.default(project_root=args.project_root)
    svi_cfg = config.svi
    net_cfg = config.network

    if args.max_iterations is not None:
        svi_cfg = replace(svi_cfg, max_iterations=args.max_iterations)
    if args.posterior_samples is not None:
        svi_cfg = replace(svi_cfg, posterior_samples=args.posterior_samples)
    if args.device is not None:
        svi_cfg = replace(svi_cfg, device=args.device)
    if args.observed_genes_only:
        net_cfg = replace(net_cfg, restrict_to_observed_genes=True)
    if args.min_component_metabolites is not None:
        net_cfg = replace(net_cfg, min_metabolites_per_component=args.min_component_metabolites)

    config = replace(config, svi=svi_cfg, network=net_cfg)
    layer_arg = None if args.counts_layer == "X_log1p" else args.counts_layer
    result = KineticsPipeline(config).run_h5ad(
        input_h5ad=args.input,
        patient_col=args.patient_col,
        celltype_col=args.celltype_col,
        output_dir=args.out,
        counts_layer=layer_arg,
        seed=args.seed,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def _summarize(args) -> int:
    output_dir = Path(args.input).expanduser().resolve()
    summary_path = output_dir / "pipeline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing pipeline summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    effective_path = output_dir / "posterior_kinetics_effective.csv"
    effective = pd.read_csv(effective_path)
    payload = {
        "summary_file": str(summary_path),
        "effective_file": str(effective_path),
        "flux_constrained": bool(summary.get("flux_constrained", False)),
        "tasks": summary.get("tasks", []),
        "parameters": sorted(effective["parameter"].dropna().astype(str).unique()),
        "n_effective_rows": int(len(effective)),
        "n_contexts": int(summary.get("n_contexts", 0)),
        "n_reactions": int(summary.get("n_reactions", 0)),
        "network": summary.get("network", {}),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
