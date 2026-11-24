import numpy as np
import pandas as pd

from enzcast_isokin.config import SVIConfig
from enzcast_isokin.isokin import (
    IsoKinModel,
    build_isoform_map,
    compute_complex_rate_limit,
    compute_effective_kinetics,
    decompose_gpr,
    logsumexp_base10,
)
from enzcast_isokin.isokin.gpr import evaluate_gpr, parse_gpr_rule


def test_decompose_gpr_and_softmin_complex_evaluation():
    assert decompose_gpr("HK1 or HK2") == (("HK1",), ("HK2",))
    assert decompose_gpr("SDHA and SDHB") == (("SDHA", "SDHB"),)
    assert decompose_gpr("(A or B) and C") == (("A", "C"), ("B", "C"))
    assert decompose_gpr("A or B or C", max_isoforms=2) == (("A",), ("B",))

    node = parse_gpr_rule("SDHA and SDHB")
    softmin_val = evaluate_gpr(node, {"SDHA": 1.0, "SDHB": 10.0}, temperature=1.0)
    assert softmin_val < 1.01

    iso_map = build_isoform_map(
        reaction_ids=["R_OR", "R_AND"],
        gpr_rules={"R_OR": "HK1 or HK2", "R_AND": "SDHA and SDHB and SDHC"},
        max_isoforms=4,
    )
    assert iso_map["n_multi_iso"] == 1
    assert iso_map["n_complex_branches"] == 1
    assert iso_map["isoform_genes"][("R_AND", 0)] == ["SDHA", "SDHB", "SDHC"]


def test_logsumexp_base10_and_effective_kinetics_linear_scale_weighting():
    values = np.array([0.0, np.log10(9.0)])
    assert np.isclose(logsumexp_base10(values), 1.0)
    assert np.isneginf(logsumexp_base10(values, mask=np.array([False, False])))

    theta_iso = np.array([[[-1.0, 0.0, 2.0, -2.0], [-0.5, 1.0, 4.0, -1.5]]])
    log10_expression = np.array([[[0.0, np.log10(9.0)]]])
    mask = np.array([[True, True]])

    effective = compute_effective_kinetics(theta_iso, log10_expression, mask)
    expected = np.array(
        [
            np.log10(0.1 * 10**-1.0 + 0.9 * 10**-0.5),
            np.log10(0.1 * 10**0.0 + 0.9 * 10**1.0),
            np.log10(0.1 * 10**2.0 + 0.9 * 10**4.0),
            np.log10(0.1 * 10**-2.0 + 0.9 * 10**-1.5),
        ]
    )
    np.testing.assert_allclose(effective[0, 0], expected)


def test_complex_rate_limit_helper():
    log10_expression = np.array([[[4.0, 1.0, 3.0]]], dtype=float)
    isoform_mask = np.array([[True, True, True]], dtype=bool)
    complex_index = [[(0, 1), (2,)]]

    limited, complex_mask = compute_complex_rate_limit(log10_expression, complex_index, isoform_mask)
    np.testing.assert_allclose(limited, np.array([[[1.0, 3.0]]]))
    np.testing.assert_array_equal(complex_mask, np.array([[True, True]]))


def test_isokin_model_runs_4task_stoichiometric_svi_and_exports_csvs(tmp_path):
    prior_mu = np.array(
        [
            [[-1.0, 1.0, 2.0, -2.0], [-0.8, 1.4, 2.2, -1.8]],
            [[-1.2, 0.8, 2.0, -2.1], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    prior_sigma = np.full_like(prior_mu, 0.25)
    log10_expression = np.array(
        [
            [[0.0, np.log10(9.0)], [0.5, -6.0]],
            [[0.8, 0.2], [0.3, -6.0]],
        ],
        dtype=float,
    )
    isoform_mask = np.array([[True, True], [True, False]], dtype=bool)
    stoichiometry = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

    prior_prov = pd.DataFrame(
        [
            {
                "reaction_id": "RXN1",
                "isoform_idx": 0,
                "isoform_genes": "HK1",
                "parameter": "kcat",
                "prior_mean_log10": 1.0,
                "prior_std_log10": 0.25,
                "prior_tier": "exact",
                "source_checkpoint": "multi_task",
            }
        ]
    )

    config = SVIConfig(max_iterations=3, posterior_samples=4, guide_rank=4)
    result = IsoKinModel(config=config).fit(
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        log10_expression=log10_expression,
        isoform_mask=isoform_mask,
        stoichiometry=stoichiometry,
        reaction_ids=["RXN1", "RXN2"],
        prior_provenance=prior_prov,
        seed=7,
    )

    assert result.isoform_samples.shape == (4, 2, 2, 4)
    assert result.effective_mean.shape == (2, 2, 4)
    assert result.effective_sd.shape == (2, 2, 4)
    assert result.tasks == ("km", "kcat", "kcat_km", "ki")
    assert result.convergence["flux_constrained"] is True
    assert result.convergence["guide"] == "AutoLowRankMultivariateNormal"
    assert result.convergence["iterations"] == 3

    written = result.export_csvs(
        tmp_path,
        context_ids=["P1|Tumor", "P1|Normal"],
        reaction_ids=["RXN1", "RXN2"],
        isoform_genes={("RXN1", 0): ["HK1"], ("RXN1", 1): ["HK2"], ("RXN2", 0): ["PDHA1", "PDHB"]},
    )
    expected_files = {
        "posterior_kinetics_isoform.csv",
        "posterior_kinetics_effective.csv",
        "omega_saturation.csv",
        "s_base_substrate.csv",
        "variance_decomposition.csv",
        "pipeline_summary.json",
    }
    assert {p.name for p in written} == expected_files

    iso_df = pd.read_csv(tmp_path / "posterior_kinetics_isoform.csv")
    eff_df = pd.read_csv(tmp_path / "posterior_kinetics_effective.csv")
    omega_df = pd.read_csv(tmp_path / "omega_saturation.csv")
    sbase_df = pd.read_csv(tmp_path / "s_base_substrate.csv")
    var_df = pd.read_csv(tmp_path / "variance_decomposition.csv")

    assert set(iso_df["branch_type"]) == {"single_gene", "subunit_complex"}
    assert "PDHA1+PDHB" in set(iso_df["isoform_id"])
    assert set(eff_df["parameter"]) == {"km", "kcat", "kcat_km", "ki"}
    assert len(omega_df) == 2 and np.isfinite(omega_df["omega_mean"]).all()
    assert len(sbase_df) == 2 and np.isfinite(sbase_df["s_base_mean"]).all()
    assert len(var_df) == 8
