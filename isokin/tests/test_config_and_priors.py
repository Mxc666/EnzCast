from pathlib import Path

import numpy as np
import pandas as pd

from enzcast_isokin import PipelineConfig
from enzcast_isokin.artifacts import ArtifactRegistry
from enzcast_isokin.enzcast import PriorTable, assemble_isoform_priors_from_csvs, denormalize_prediction
from enzcast_isokin.isokin.gpr import build_isoform_map


IDENTITY = {
    "UniProt_ID": "P1",
    "Model_Input_Sequence": "MPEPTIDE",
    "SMILES": "CCO",
    "pH": 7.0,
    "Temperature_C": 37.0,
    "Organism": "Homo sapiens",
    "EnzymeType": "wild-type",
    "EC_Number": "1.1.1.1",
    "reaction_id": "RXN1",
    "gene_name": "GENE1",
    "subsystem": "Toy metabolism",
}


def write_prior(path: Path, extra: dict[str, float]) -> None:
    pd.DataFrame([{**IDENTITY, **extra}]).to_csv(path, index=False)


def test_default_config_uses_injected_root_and_stoichiometric_svi_defaults(tmp_path):
    config = PipelineConfig.default(project_root=tmp_path)

    assert config.public_method_names == ("EnzCast", "KinBench", "IsoKin")
    assert config.default_condition.ph == 7.0
    assert config.default_condition.temperature_c == 37.0
    assert config.default_condition.organism == "Homo sapiens"
    assert config.default_condition.enzyme_type == "wild-type"
    assert config.svi.optimizer == "ClippedAdam"
    assert config.svi.learning_rate == 0.001
    assert config.svi.learning_rate_decay == 0.9999
    assert config.svi.clip_norm == 10.0
    assert config.svi.max_iterations == 20000
    assert config.svi.posterior_samples == 2000
    assert config.posterior_tasks == ("km", "kcat", "kcat_km", "ki")
    assert str(config.project_root) == str(tmp_path)


def test_artifact_registry_routes_tasks_and_registers_recon3d_assets(tmp_path):
    registry = ArtifactRegistry.default(project_root=tmp_path)

    assert registry.checkpoint_family_for_task("km") == "single_task_km"
    assert registry.checkpoint_family_for_task("ki") == "single_task_ki"
    assert registry.checkpoint_family_for_task("kcat") == "multi_task"
    assert registry.checkpoint_family_for_task("kcat_km") == "multi_task"
    assert registry.get("recon3d_json").path == tmp_path / "Dataset" / "isokin" / "recon3d_model.json"
    assert registry.get("gpr_rules_json").path == tmp_path / "Dataset" / "isokin" / "recon3d_gpr_rules.json"
    assert registry.get("recon3d_inputs_csv").path == tmp_path / "Dataset" / "isokin" / "recon3d_enzcast_inputs.csv"
    assert registry.get("prior_mtl_csv").path == tmp_path / "Dataset" / "isokin" / "priors" / "enzcast_priors_multitask.csv"
    assert registry.get("checkpoint_mtl").path == tmp_path / "Model" / "fulldata_priors" / "enzcast_fulldata_multitask.pt"


def test_prior_table_and_isoform_prior_assembly_route_km_ki_kcat_correctly(tmp_path):
    mtl = tmp_path / "mtl.csv"
    st_km = tmp_path / "st_km.csv"
    st_ki = tmp_path / "st_ki.csv"
    write_prior(
        mtl,
        {
            "km_mean_log10": -9.0,
            "km_std_log10": 9.0,
            "kcat_mean_log10": 1.0,
            "kcat_std_log10": 0.2,
            "kcat_km_mean_log10": 2.0,
            "kcat_km_std_log10": 0.3,
            "ki_mean_log10": -8.0,
            "ki_std_log10": 8.0,
        },
    )
    write_prior(st_km, {"km_mean_log10": -1.5, "km_std_log10": 0.4})
    write_prior(st_ki, {"ki_mean_log10": -2.5, "ki_std_log10": 0.5})

    priors = PriorTable.from_precomputed_csvs(mtl_csv=mtl, st_km_csv=st_km, st_ki_csv=st_ki)
    long_df = priors.to_frame().sort_values("parameter").reset_index(drop=True)

    assert set(long_df["parameter"]) == {"ki", "km", "kcat", "kcat_km"}
    assert long_df.loc[long_df["parameter"] == "km", "mean_log10"].item() == -1.5
    assert long_df.loc[long_df["parameter"] == "ki", "mean_log10"].item() == -2.5
    assert long_df.loc[long_df["parameter"] == "kcat", "mean_log10"].item() == 1.0
    assert long_df.loc[long_df["parameter"] == "kcat_km", "mean_log10"].item() == 2.0

    iso_map = build_isoform_map(
        reaction_ids=["RXN1", "RXN2"],
        gpr_rules={"RXN1": "GENE1 or (GENE1 and GENE_UNSEEN)", "RXN2": "GENE_OTHER"},
        max_isoforms=4,
    )
    assembled = assemble_isoform_priors_from_csvs(
        mtl_csv=mtl,
        st_km_csv=st_km,
        st_ki_csv=st_ki,
        reaction_ids=["RXN1", "RXN2"],
        isoform_map=iso_map,
        max_isoforms=4,
    )
    np.testing.assert_allclose(assembled["prior_mu_iso"][0, 0], np.array([-1.5, 1.0, 2.0, -2.5]))
    assert int(assembled["match_type"][0, 0]) == 1
    assert int(assembled["match_type"][1, 0]) == 3


def test_denormalize_prediction_matches_reference_formula():
    mu, sigma = denormalize_prediction(
        mu_norm=np.array([0.0, 2.0]),
        logvar_norm=np.array([0.0, np.log(4.0)]),
        task_mean=10.0,
        task_std=0.5,
    )

    np.testing.assert_allclose(mu, np.array([10.0, 11.0]))
    np.testing.assert_allclose(sigma, np.array([0.5, 1.0]))
