# Parameter Reference

Default hyperparameters in `enzcast_isokin.config` and `enzcast_isokin.stoichiometric_model`.

## Biological Prior Condition Defaults (`ConditionDefaults`)

| Parameter | Default | Description |
|---|---:|---|
| `ph` | `7.0` | Physiological pH used for default human Recon3D priors |
| `temperature_c` | `37.0` | Physiological temperature (`°C`) |
| `organism` | `Homo sapiens` | Default organism vocabulary entry |
| `enzyme_type` | `wild-type` | Default variant category |

## Stoichiometric Network & GPR Expression (`NetworkConfig`, `ExpressionConfig`)

| Parameter | Default | Description |
|---|---:|---|
| `min_reactions_per_metabolite` | `2` | Iterative pruning threshold removing dead-end internal metabolites |
| `min_metabolites_per_component` | `3` | Minimum internal metabolites required per retained connected component |
| `max_isoforms` | `10` | Maximum disjunctive `OR` isoform branches (`I_max`) per reaction |
| `softmin_temperature` | `1.0` | Smooth soft-min temperature `T` for `AND` multi-subunit complexes |
| `kappa` | `1.0` | Linear expression-to-abundance scaling constant |
| `pseudocount` | `1e-6` | Floor added to linear isoform abundance prior to `log10` |
| `detect_threshold` | `0.05` | Soft expression detection midpoint for smooth flux gating |
| `detect_steepness` | `10.0` | Logistic steepness of smooth detection gate |

## Pyro Variational Inference (`SVIConfig`)

| Parameter | Default | Description |
|---|---:|---|
| `optimizer` | `ClippedAdam` | Pyro stochastic variational optimizer |
| `guide` | `AutoLowRankMultivariateNormal` | Variational posterior family |
| `guide_rank` | `10` | Low-rank covariance rank (automatically capped below latent dimension on tiny networks) |
| `learning_rate` | `0.001` | Initial SVI step size |
| `learning_rate_decay` | `0.9999` | Multiplicative step-size decay (`lrd`) |
| `clip_norm` | `10.0` | Gradient norm clipping threshold |
| `max_iterations` | `20000` | Default SVI optimization steps |
| `posterior_samples` | `2000` | Posterior Monte Carlo draws exported for summary tables |
| `elbo_particles` | `1` | ELBO Monte Carlo particles per step |
| `alpha_d` | `0.30` | Base steady-state flux-balance residual tolerance (`sigma_d = alpha_d * sqrt(deg_m)`) |
| `consistency_weight` | `0.10` | Thermodynamic consistency weight linking `kcat_km`, `kcat`, and `km` |
| `sigma_base_prior` | `0.50` | Prior standard deviation on baseline substrate `s_base` (`log10 mM`) |
| `sigma_s_global_prior` | `0.20` | Global hierarchical scale prior for context substrate shifts |
| `sigma_s_context_prior` | `0.15` | Per-context substrate deviation scale prior |
| `scale_anchor_sigma` | `0.02` | Mean-zero identifiability anchor across `s_base` |
| `missing_sigma_inflate` | `1.5` | Prior standard deviation multiplier for fallback isoform branches |

## Task Routing Policy

| Parameter | EnzCast Checkpoint Family | Exported in IsoKin Posteriors | Primary Biological Headline |
|---|---|---|---|
| `km` | Single-task `enzcast_fulldata_km.pt` | Yes (`isoform` & `effective`) | Auxiliary |
| `kcat` | Multi-task `enzcast_fulldata_multitask.pt` | Yes (`isoform` & `effective`) | Primary (`is_primary = True`) |
| `kcat_km` | Multi-task `enzcast_fulldata_multitask.pt` | Yes (`isoform` & `effective`) | Primary (`is_primary = True`) |
| `ki` | Single-task `enzcast_fulldata_ki.pt` | Yes (`isoform` & `effective`) | Auxiliary |
