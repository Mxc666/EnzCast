from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    key: str
    path: Path
    description: str
    required: bool = True
    remote_uri: str | None = None


class ArtifactRegistry:
    _TASK_TO_CHECKPOINT_FAMILY = {
        "km": "single_task_km",
        "ki": "single_task_ki",
        "kcat": "multi_task",
        "kcat_km": "multi_task",
    }

    def __init__(self, artifacts: dict[str, Artifact]) -> None:
        self._artifacts = dict(artifacts)

    @classmethod
    def default(cls, project_root: str | Path) -> "ArtifactRegistry":
        root = Path(project_root).expanduser().resolve()
        scaffold_dir = root / "Dataset" / "isokin"
        prior_dir = scaffold_dir / "priors"
        model_dir = root / "Model" / "fulldata_priors"
        artifacts = {
            "recon3d_json": Artifact(
                "recon3d_json",
                scaffold_dir / "recon3d_model.json",
                "Recon3D metabolic network JSON",
            ),
            "gpr_rules_json": Artifact(
                "gpr_rules_json",
                scaffold_dir / "recon3d_gpr_rules.json",
                "Recon3D gene-protein-reaction (GPR) rules JSON",
            ),
            "prior_mtl_csv": Artifact(
                "prior_mtl_csv",
                prior_dir / "enzcast_priors_multitask.csv",
                "Precomputed multi-task EnzCast prior CSV",
            ),
            "prior_st_km_csv": Artifact(
                "prior_st_km_csv",
                prior_dir / "enzcast_priors_km.csv",
                "Precomputed single-task Km EnzCast prior CSV",
            ),
            "prior_st_ki_csv": Artifact(
                "prior_st_ki_csv",
                prior_dir / "enzcast_priors_ki.csv",
                "Precomputed single-task Ki EnzCast prior CSV",
            ),
            "recon3d_inputs_csv": Artifact(
                "recon3d_inputs_csv",
                scaffold_dir / "recon3d_enzcast_inputs.csv",
                "Recon3D reaction-gene-substrate EnzCast input table",
                required=False,
            ),
            "checkpoint_mtl": Artifact(
                "checkpoint_mtl",
                model_dir / "enzcast_fulldata_multitask.pt",
                "Multi-task full-data checkpoint for kcat and kcat/Km priors",
                required=False,
            ),
            "checkpoint_st_km": Artifact(
                "checkpoint_st_km",
                model_dir / "enzcast_fulldata_km.pt",
                "Single-task full-data Km checkpoint",
                required=False,
            ),
            "checkpoint_st_ki": Artifact(
                "checkpoint_st_ki",
                model_dir / "enzcast_fulldata_ki.pt",
                "Single-task full-data Ki checkpoint",
                required=False,
            ),
            "metadata_fulldata": Artifact(
                "metadata_fulldata",
                model_dir / "enzcast_fulldata_metadata.json",
                "Full-data model metadata, normalization statistics, and organism vocabularies",
                required=False,
            ),
        }
        return cls(artifacts)

    @classmethod
    def default_paper(cls, project_root: str | Path) -> "ArtifactRegistry":
        return cls.default(project_root)

    def get(self, key: str) -> Artifact:
        try:
            return self._artifacts[key]
        except KeyError as exc:
            known = ", ".join(sorted(self._artifacts))
            raise KeyError(f"Unknown artifact key {key!r}; known keys: {known}") from exc

    def checkpoint_family_for_task(self, task: str) -> str:
        normalized = task.lower()
        try:
            return self._TASK_TO_CHECKPOINT_FAMILY[normalized]
        except KeyError as exc:
            known = ", ".join(sorted(self._TASK_TO_CHECKPOINT_FAMILY))
            raise KeyError(f"Unknown kinetic task {task!r}; known tasks: {known}") from exc

    def as_dict(self) -> dict[str, Artifact]:
        return dict(self._artifacts)

    def manifest(self) -> list[dict[str, object]]:
        rows = []
        for key in sorted(self._artifacts):
            artifact = self._artifacts[key]
            rows.append(
                {
                    "key": artifact.key,
                    "path": str(artifact.path),
                    "description": artifact.description,
                    "required": artifact.required,
                    "exists": artifact.path.exists(),
                    "remote_uri": artifact.remote_uri,
                    "download_attempted": False,
                }
            )
        return rows
