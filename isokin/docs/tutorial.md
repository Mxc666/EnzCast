# Tutorial: From Cohort `.h5ad` to IsoKin Posterior Profiles

This guide walks through running the complete IsoKin cohort fitting workflow via CLI or Python.

## 1. Required Project Layout

Point `--project-root` to the top-level directory containing `Dataset/isokin/`:

- `Dataset/isokin/recon3d_model.json`: Recon3D metabolic stoichiometry and reaction bounds.
- `Dataset/isokin/recon3d_gpr_rules.json`: Boolean `AND`/`OR` Gene–Protein–Reaction rules.
- `Dataset/isokin/priors/enzcast_priors_multitask.csv`: Full-data EnzCast `kcat` and `kcat_km` priors.
- `Dataset/isokin/priors/enzcast_priors_km.csv`: Full-data EnzCast single-task `km` priors.
- `Dataset/isokin/priors/enzcast_priors_ki.csv`: Full-data EnzCast single-task `ki` priors.

## 2. Validate Cohort `.h5ad`

```bash
enzcast-isokin validate-h5ad \
  --input ../examples/sample_cohort.h5ad \
  --patient-col subject_id \
  --celltype-col cell_state
```

`validate-h5ad` checks required `obs` columns, non-empty context labels, expression layer availability, and gene symbol overlap with Recon3D GPR rules.

## 3. Fit Stoichiometric + GPR-Resolved Kinetic Posteriors

During `infer`, IsoKin executes:
1. **Recon3D internal stoichiometry construction**: excludes ubiquitous currency cofactors (`h`, `h2o`, `atp`, `adp`, `nad`, `nadh`, `nadp`, `nadph`, `pi`, etc.), prunes dead-end metabolites (`degree < 2`), and retains connected internal metabolic components (`S` of shape `M × R`).
2. **Recursive GPR isoform & complex decomposition**: parses each reaction's boolean GPR into disjunctive isoform branches (`OR`, up to `I_max = 10`), evaluating multi-subunit `AND` complexes via smooth soft-min (`-T · logsumexp(-v / T)`) over group-mean linear expression.
3. **Task-adaptive EnzCast prior assembly**: assigns `(mu_0, sigma_0)` per isoform branch from single-task `km`/`ki` and multi-task `kcat`/`kcat_km` priors, with reaction-average and global fallback inflation (`1.5×`) when a branch lacks a direct sequence entry.
4. **Soft flux-balanced Pyro SVI**: fits `AutoLowRankMultivariateNormal(rank=10)` with `ClippedAdam` under thermodynamic consistency (`kcat/Km ≈ kcat - Km`) and degree-scaled steady-state soft flux balance (`S · v ≈ 0`).

```bash
enzcast-isokin infer \
  --input ../examples/sample_cohort.h5ad \
  --patient-col subject_id \
  --celltype-col cell_state \
  --project-root ../.. \
  --observed-genes-only \
  --max-iterations 200 \
  --posterior-samples 200 \
  --out /tmp/isokin_demo
```

## 4. Inspect Outputs

```bash
enzcast-isokin summarize --input /tmp/isokin_demo
```

Inspect `pipeline_summary.json` to verify `"flux_constrained": true`, internal metabolite count, constrained reaction count, multi-isoform reaction count, and multi-subunit complex branch count.
