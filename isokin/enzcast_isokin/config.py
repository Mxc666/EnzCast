from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from enzcast_isokin.artifacts import ArtifactRegistry


@dataclass(frozen=True)
class DefaultCondition:
    ph: float = 7.0
    temperature_c: float = 37.0
    organism: str = "Homo sapiens"
    enzyme_type: str = "wild-type"


@dataclass(frozen=True)
class NetworkConfig:
    min_reactions_per_metabolite: int = 2
    min_metabolites_per_component: int = 3
    restrict_to_observed_genes: bool = False
    reaction_subset: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExpressionConfig:
    kappa: float = 1.0
    pseudocount: float = 1e-6
    softmin_temperature: float = 1.0
    detect_threshold: float = 0.05
    detect_steepness: float = 10.0
    target_sum_cp10k: float = 1e4


@dataclass(frozen=True)
class SVIConfig:
    optimizer: str = "ClippedAdam"
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.9999
    clip_norm: float = 10.0
    max_iterations: int = 20000
    convergence_window: int = 1000
    convergence_threshold: float = 1e-3
    guide_rank: int = 10
    posterior_samples: int = 2000
    max_isoforms: int = 10
    missing_sigma_inflate: float = 1.5
    sigma_s_global_prior: float = 0.20
    sigma_s_context_prior: float = 0.15
    sigma_base_prior: float = 0.50
    consistency_weight: float = 0.10
    ki_tight_sigma: float = 0.20
    alpha_d: float = 0.30
    scale_anchor_sigma: float = 0.02
    device: str = "auto"


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    default_condition: DefaultCondition
    svi: SVIConfig
    artifacts: ArtifactRegistry
    network: NetworkConfig = field(default_factory=NetworkConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    public_method_names: tuple[str, str, str] = ("EnzCast", "KinBench", "IsoKin")
    posterior_tasks: tuple[str, str, str, str] = ("km", "kcat", "kcat_km", "ki")
    primary_posterior_tasks: tuple[str, str] = ("kcat", "kcat_km")
    auxiliary_posterior_tasks: tuple[str, str] = ("km", "ki")

    @classmethod
    def default(cls, project_root: str | Path) -> "PipelineConfig":
        root = Path(project_root).expanduser().resolve()
        return cls(
            project_root=root,
            default_condition=DefaultCondition(),
            svi=SVIConfig(),
            artifacts=ArtifactRegistry.default(root),
            network=NetworkConfig(),
            expression=ExpressionConfig(),
        )

    @classmethod
    def default_paper(cls, project_root: str | Path) -> "PipelineConfig":
        return cls.default(project_root)
