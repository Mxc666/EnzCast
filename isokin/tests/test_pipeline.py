from dataclasses import replace
import json

import anndata as ad
import numpy as np
import pandas as pd

from enzcast_isokin import KineticsPipeline, PipelineConfig


def write_prior(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def prior_row(gene, reaction_id, **values):
    base = {
        "inference_id": f"R3D_{reaction_id}_{gene}",
        "UniProt_ID": f"UP_{gene}",
        "Model_Input_Sequence": "MPEPTIDE",
        "SMILES": "CCO",
        "pH": 7.0,
        "Temperature_C": 37.0,
        "Organism": "Homo sapiens",
        "EnzymeType": "wild-type",
        "EC_Number": "1.1.1.1",
        "reaction_id": reaction_id,
        "gene_name": gene,
        "subsystem": "Central carbon",
    }
    base.update(values)
    return base


def make_project_root(tmp_path):
    config = PipelineConfig.default(project_root=tmp_path)
    config = replace(
        config,
        network=replace(config.network, min_metabolites_per_component=2, restrict_to_observed_genes=True),
    )

    recon3d_path = config.artifacts.get("recon3d_json").path
    gpr_path = config.artifacts.get("gpr_rules_json").path
    mtl = config.artifacts.get("prior_mtl_csv").path
    st_km = config.artifacts.get("prior_st_km_csv").path
    st_ki = config.artifacts.get("prior_st_ki_csv").path

    recon3d_path.parent.mkdir(parents=True, exist_ok=True)
    recon3d_model = {
        "metabolites": [
            {"id": "glc_c", "name": "Glucose", "compartment": "c"},
            {"id": "g6p_c", "name": "Glucose 6-phosphate", "compartment": "c"},
            {"id": "f6p_c", "name": "Fructose 6-phosphate", "compartment": "c"},
            {"id": "atp_c", "name": "ATP", "compartment": "c"},
        ],
        "reactions": [
            {
                "id": "RXN1",
                "name": "Hexokinase",
                "subsystem": "Glycolysis",
                "lower_bound": 0.0,
                "upper_bound": 1000.0,
                "metabolites": {"glc_c": -1.0, "atp_c": -1.0, "g6p_c": 1.0},
            },
            {
                "id": "RXN2",
                "name": "Glucose-6-phosphate isomerase",
                "subsystem": "Glycolysis",
                "lower_bound": -1000.0,
                "upper_bound": 1000.0,
                "metabolites": {"g6p_c": -1.0, "f6p_c": 1.0},
            },
            {
                "id": "RXN3",
                "name": "Complex step",
                "subsystem": "Glycolysis",
                "lower_bound": 0.0,
                "upper_bound": 1000.0,
                "metabolites": {"f6p_c": -1.0, "glc_c": 1.0},
            },
            {
                "id": "RXN_UNOBS",
                "name": "Unobserved side reaction",
                "subsystem": "Other",
                "lower_bound": 0.0,
                "upper_bound": 1000.0,
                "metabolites": {"g6p_c": -1.0, "f6p_c": 1.0},
            },
        ],
    }
    recon3d_path.write_text(json.dumps(recon3d_model, indent=2) + "\n", encoding="utf-8")

    gpr_rules = {
        "RXN1": "GENE1 or GENE2",
        "RXN2": "GENE5",
        "RXN3": "GENE3 and GENE4",
        "RXN_UNOBS": "GENE_MISSING",
    }
    gpr_path.write_text(json.dumps(gpr_rules, indent=2) + "\n", encoding="utf-8")

    rows = [
        prior_row("GENE1", "RXN1", km_mean_log10=-1.0, km_std_log10=0.2, kcat_mean_log10=1.0, kcat_std_log10=0.2, kcat_km_mean_log10=2.0, kcat_km_std_log10=0.3, ki_mean_log10=-2.0, ki_std_log10=0.4),
        prior_row("GENE2", "RXN1", km_mean_log10=-1.2, km_std_log10=0.2, kcat_mean_log10=1.3, kcat_std_log10=0.2, kcat_km_mean_log10=2.5, kcat_km_std_log10=0.3, ki_mean_log10=-2.2, ki_std_log10=0.4),
        prior_row("GENE5", "RXN2", km_mean_log10=-0.8, km_std_log10=0.2, kcat_mean_log10=1.5, kcat_std_log10=0.2, kcat_km_mean_log10=2.3, kcat_km_std_log10=0.3, ki_mean_log10=-1.9, ki_std_log10=0.4),
        prior_row("GENE3", "RXN3", km_mean_log10=-1.1, km_std_log10=0.2, kcat_mean_log10=0.9, kcat_std_log10=0.2, kcat_km_mean_log10=2.0, kcat_km_std_log10=0.3, ki_mean_log10=-2.1, ki_std_log10=0.4),
        prior_row("GENE4", "RXN3", km_mean_log10=-0.9, km_std_log10=0.2, kcat_mean_log10=1.1, kcat_std_log10=0.2, kcat_km_mean_log10=2.0, kcat_km_std_log10=0.3, ki_mean_log10=-2.0, ki_std_log10=0.4),
        prior_row("GENE_MISSING", "RXN_UNOBS", km_mean_log10=-1.0, km_std_log10=0.2, kcat_mean_log10=1.0, kcat_std_log10=0.2, kcat_km_mean_log10=2.0, kcat_km_std_log10=0.3, ki_mean_log10=-2.0, ki_std_log10=0.4),
    ]
    write_prior(mtl, rows)
    write_prior(st_km, rows)
    write_prior(st_ki, rows)
    return config


def make_h5ad(path, lowercase: bool = False):
    obs = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "cell_type": ["Tumor", "Normal", "Tumor", "Normal"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    if lowercase:
        genes = [g.lower() for g in genes]
    var = pd.DataFrame(index=genes)
    counts = np.array(
        [
            [5.0, 1.0, 4.0, 2.0, 3.0],
            [1.0, 4.0, 3.0, 3.0, 2.0],
            [6.0, 0.0, 5.0, 1.0, 4.0],
            [2.0, 3.0, 2.0, 4.0, 2.0],
        ],
        dtype=float,
    )
    adata = ad.AnnData(X=np.log1p(counts), obs=obs, var=var, layers={"counts": counts})
    adata.write_h5ad(path)


def test_kinetics_pipeline_runs_stoichiometric_gpr_fitting_with_flux_constraint(tmp_path):
    config = make_project_root(tmp_path / "project")
    config = replace(config, svi=replace(config.svi, max_iterations=2, posterior_samples=4, guide_rank=4))
    input_h5ad = tmp_path / "cohort.h5ad"
    output_dir = tmp_path / "out"
    make_h5ad(input_h5ad)

    result = KineticsPipeline(config).run_h5ad(
        input_h5ad=input_h5ad,
        patient_col="patient_id",
        celltype_col="cell_type",
        output_dir=output_dir,
        seed=11,
    )

    assert result.summary["flux_constrained"] is True
    assert result.summary["retained_reactions"] == ["RXN1", "RXN2", "RXN3"]
    assert result.summary["network"]["n_internal_metabolites"] == 3
    assert result.summary["network"]["n_multi_isoform_reactions"] == 1
    assert result.summary["network"]["n_subunit_complex_branches"] == 1

    isoform = pd.read_csv(output_dir / "posterior_kinetics_isoform.csv")
    effective = pd.read_csv(output_dir / "posterior_kinetics_effective.csv")
    omega = pd.read_csv(output_dir / "omega_saturation.csv")
    sbase = pd.read_csv(output_dir / "s_base_substrate.csv")

    assert bool(effective["flux_constrained"].iloc[0]) is True
    assert set(effective["parameter"]) == {"km", "kcat", "kcat_km", "ki"}
    assert "GENE3+GENE4" in set(isoform["isoform_id"])
    assert "subunit_complex" in set(isoform["branch_type"])
    assert len(omega) == 3 and np.isfinite(omega["omega_mean"]).all()
    assert len(sbase) == 3 and np.isfinite(sbase["s_base_mean"]).all()


def test_pipeline_resolves_case_insensitive_gene_symbols_and_excludes_unobserved_reactions(tmp_path):
    config = make_project_root(tmp_path / "project")
    config = replace(config, svi=replace(config.svi, max_iterations=1, posterior_samples=2, guide_rank=4))
    input_h5ad = tmp_path / "cohort_lower.h5ad"
    output_dir = tmp_path / "out"
    make_h5ad(input_h5ad, lowercase=True)

    result = KineticsPipeline(config).run_h5ad(
        input_h5ad=input_h5ad,
        patient_col="patient_id",
        celltype_col="cell_type",
        output_dir=output_dir,
        seed=19,
    )

    assert result.summary["retained_reactions"] == ["RXN1", "RXN2", "RXN3"]
    assert "RXN_UNOBS" not in result.summary["retained_reactions"]
