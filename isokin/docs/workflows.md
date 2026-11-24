# Workflow Patterns

All cohort workflows share the same `enzcast-isokin` CLI and Python interface.

## 1. Transcriptome-Wide Single-Cell or Pseudobulk Cohort

For full scRNA-seq / snRNA-seq cohorts where `obs` defines subject IDs and cell-type or condition strata:

```bash
enzcast-isokin infer \
  --input /path/to/cohort.h5ad \
  --patient-col patient_id \
  --celltype-col cell_type \
  --project-root /path/to/EnzCast-Release \
  --device auto \
  --out /path/to/cohort_posteriors
```

- If `adata.X` is already `log1p`-normalized, `--counts-layer auto` uses `np.expm1(adata.X)` directly.
- If `adata.layers["counts"]` stores raw UMI counts (and `adata.X` is unnormalized), `--counts-layer auto` normalizes each cell to `10,000` counts (`CP10K`) followed by `log1p`. You may also force a specific layer via `--counts-layer counts` or `--counts-layer X_log1p`.

## 2. Targeted Metabolic Panel or Compact Demo Cohort

When an `.h5ad` file profiles only a subset of metabolic genes (such as `Code/examples/sample_cohort.h5ad`), pass `--observed-genes-only` so candidate reactions are restricted to reactions with observed genes before building connected stoichiometric components:

```bash
enzcast-isokin infer \
  --input Code/examples/sample_cohort.h5ad \
  --patient-col subject_id \
  --celltype-col cell_state \
  --project-root . \
  --observed-genes-only \
  --out /tmp/panel_posteriors
```

## 3. Custom EnzCast Priors for Updated Conditions or Enzymes

To regenerate or customize Recon3D priors (for example, at an alternative physiological pH or temperature):
1. Run `Code/enzcast/predict.py --suite fulldata` on `Dataset/isokin/recon3d_enzcast_inputs.csv` (optionally passing `--ph` and `--temperature-c`).
2. Save updated prior tables under a custom project root (`Dataset/isokin/priors/enzcast_priors_{multitask,km,ki}.csv`) with columns `reaction_id`, `gene_name`, and `{task}_{mean,std}_log10`.
3. Pass that directory to `enzcast-isokin infer --project-root /path/to/custom_root`.
