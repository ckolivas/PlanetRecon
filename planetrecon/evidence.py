"""W00 evidence manifests for archived R9 tables and the current operators."""

from __future__ import annotations

from pathlib import Path

from planetrecon import constants as C
from planetrecon.provenance import experiment_manifest, load_manifest, write_manifest
from planetrecon.rank import ranking_config_hash


def r9_historical_manifest() -> dict:
    prompt2 = [
        "results/prompt2/classification_eval_feature.json",
        "results/prompt2/classification_eval_bland.json",
        "results/prompt2/classification_dev_feature.json",
        "results/prompt2/classification_dev_bland.json",
        "results/prompt2/frozen_regularisation.json",
        "results/prompt2/summary_eval.json",
        "results/prompt2/summary_dev.json",
        "results/prompt2/ranking_hash.txt",
    ]
    q2 = [
        "results/q2/closure_eval_feature.json",
        "results/q2/closure_eval_bland.json",
        "results/q2/closure_dev_feature.json",
        "results/q2/closure_dev_bland.json",
        "results/q2/frozen_prior.json",
        "results/q2/holdout_dev_seed-01001_feature.json",
        "results/q2/summary_eval.json",
        "results/q2/summary_dev.json",
    ]
    return experiment_manifest(
        "r9-historical-bundle",
        status="valid",
        protocol=(
            "Frozen R9 Gate-1 and Q2 tables. Archived in place by W00; files are "
            "not regenerated and must not be relabelled as newly validated data."
        ),
        seed_coverage={
            "development": list(C.DEV_SEEDS),
            "evaluation": list(C.EVAL_SEEDS),
            "q2_scope_dr0": list(C.Q2_SCOPE_DR0),
        },
        tolerances={
            "E1_E2A0_EH_REL_TOL": C.E1_E2A0_EH_REL_TOL,
            "Q2_CLOSURE_TARGET": C.Q2_CLOSURE_TARGET,
            "RH_ILLCONDITIONED": C.RH_ILLCONDITIONED,
        },
        inputs=[{"kind": "hdf5-truth", "schema": C.SCHEMA_NAME, "revision": C.REVISION}],
        results=[
            {
                "path": path,
                "status": "diagnostic" if "bland" in path else "valid",
                "gate_passed": False,
            }
            for path in prompt2 + q2
        ],
        notes=(
            "Gate-1 feature G1 is strong only at D/r0=4 and is oracle-sensitive. "
            "Q2 evaluation feature-rich median C=0.36388457737905266 < 0.40, so Q3 "
            "does not start. Bland tables are high-band ill-conditioned diagnostics."
        ),
        extra={
            "archived": True,
            "code_baseline": "d1b600a",
            "ranking_hash": ranking_config_hash(),
            "hdf5_revision": C.REVISION,
            "roadmap_revision": C.ROADMAP_REVISION,
        },
    )


def w00_w01_manifest() -> dict:
    return experiment_manifest(
        "w00-w01-operator-audit",
        status="diagnostic",
        protocol=(
            "W00 versioned evidence and test tiers; W01 Dykstra positivity/support "
            "projection, forward/adjoint audit, noise-approximation quantification, "
            "and method-certificate fingerprints. CPU-only, 8 threads. No Q3 and no "
            "full seed-family rerun."
        ),
        seed_coverage={"unit_fixtures": True, "scientific_families": False},
        tolerances={
            "DYKSTRA_TOL": C.DYKSTRA_TOL,
            "FEASIBLE_POS_TOL": C.FEASIBLE_POS_TOL,
            "FEASIBLE_SUPPORT_TOL": C.FEASIBLE_SUPPORT_TOL,
            "RANK_LOFREQ_JACCARD_MIN": C.RANK_LOFREQ_JACCARD_MIN,
            "RANK_LOFREQ_SPEARMAN_MIN": C.RANK_LOFREQ_SPEARMAN_MIN,
        },
        results=[
            {
                "path": "tests/test_w00.py",
                "status": "diagnostic",
                "kind": "unit",
            },
            {
                "path": "tests/test_w01.py",
                "status": "diagnostic",
                "kind": "unit",
            },
        ],
        notes=(
            "Estimator operator version is 1.1 (Dykstra). Simulator operator "
            "version remains 1.0. Historical R9 result files are unchanged."
        ),
        extra={
            "archived": False,
            "cpu_threads": C.DEFAULT_CPU_THREADS,
            "device": "cpu",
        },
    )


def write_evidence_manifests(results_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    dest = results_dir / "manifests"
    dest.mkdir(parents=True, exist_ok=True)
    written = [
        write_manifest(dest / "r9-historical.json", r9_historical_manifest()),
        write_manifest(dest / "w00-w01.json", w00_w01_manifest()),
    ]
    for path in written:
        load_manifest(path)
    return written
