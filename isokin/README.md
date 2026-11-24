# EnzCast + IsoKin (`enzcast-isokin`)

Installable Python package and CLI for fitting **IsoKin** isoform-resolved and reaction-level effective kinetic posteriors on transcriptomic cohorts (`.h5ad`) under Recon3D stoichiometric flux-balance constraints.

## Install

```bash
python -m pip install -e /path/to/EnzCast-Release/Code/isokin
```

The package expects the Recon3D metabolic network, GPR rules, and EnzCast prior CSVs under `--project-root` (`Dataset/isokin/`).

## CLI

```bash
# 1. Validate cohort AnnData schema and GPR gene coverage
enzcast-isokin validate-h5ad \
  --input /path/to/cohort.h5ad \
  --patient-col patient_id \
  --celltype-col cell_type

# 2. Export assembled long-form EnzCast priors
enzcast-isokin make-priors \
  --project-root /path/to/EnzCast-Release \
  --out /tmp/priors_long.csv

# 3. Fit stoichiometric + GPR-resolved posterior kinetic profiles
enzcast-isokin infer \
  --input /path/to/cohort.h5ad \
  --patient-col patient_id \
  --celltype-col cell_type \
  --project-root /path/to/EnzCast-Release \
  --out /path/to/isokin_output

# 4. Summarize posterior outputs
enzcast-isokin summarize --input /path/to/isokin_output
```

When running on targeted gene panels or compact examples rather than transcriptome-wide cohorts, pass `--observed-genes-only` so network construction restricts candidate reactions to those with at least one gene present in `adata.var_names`.

## Python API

```python
from dataclasses import replace
from enzcast_isokin import KineticsPipeline, PipelineConfig

config = PipelineConfig.default(project_root="/path/to/EnzCast-Release")
config = replace(config, svi=replace(config.svi, max_iterations=500, posterior_samples=500))

result = KineticsPipeline(config).run_h5ad(
    input_h5ad="/path/to/cohort.h5ad",
    patient_col="patient_id",
    celltype_col="cell_type",
    output_dir="/path/to/isokin_output",
)
```

## Input AnnData Requirements

- `obs[patient_col]`: subject, donor, or sample identifier.
- `obs[celltype_col]`: cell type, state, or condition stratum.
- `var_names` (or `var["GeneSymbol"]` / `var["symbol"]`): HGNC gene symbols overlapping Recon3D GPR rules.
- Expression values (`--counts-layer auto` by default): accepts either `log1p`-normalized `adata.X` or raw UMI counts in `adata.layers["counts"]` / `adata.X` (automatically library-size normalized to CP10K + `log1p`).

## Output Files

`enzcast-isokin infer` writes:

- `posterior_kinetics_isoform.csv`: Posterior `mean_log10`, `sd_log10`, `q05_log10`, and `q95_log10` per `(reaction_id, isoform_idx, parameter)` across `km`, `kcat`, `kcat_km`, and `ki`, labeled with `isoform_id` (e.g. `PDHA1+PDHB+DLAT+DLD` for `AND` complexes) and `branch_type` (`single_gene` or `subunit_complex`).
- `posterior_kinetics_effective.csv`: Context-specific reaction-level effective kinetic posteriors (`km`, `kcat`, `kcat_km`, `ki`) computed by linear-scale isoform expression weighting across posterior draws.
- `omega_saturation.csv`: Posterior saturation latent summaries per constrained reaction.
- `s_base_substrate.csv`: Posterior baseline substrate concentration (`log10 mM`) per constrained reaction.
- `variance_decomposition.csv`: Decomposition of across-context effective kinetic variance into isoform-expression switching (`frac_isoform`) versus posterior kinetic uncertainty (`frac_kinetic`).
- `pipeline_summary.json`: Stoichiometric matrix dimensions, internal metabolite and connected-component counts, multi-isoform and subunit-complex branch counts, `flux_constrained` flag, and SVI convergence diagnostics.

## Additional Documentation

- `docs/tutorial.md`: End-to-end walkthrough from `.h5ad` to posterior kinetic tables.
- `docs/workflows.md`: Transcriptome-wide cohorts, targeted panels, and custom prior tables.
- `docs/parameters.md`: Network, GPR expression, and Pyro SVI default hyperparameters.
