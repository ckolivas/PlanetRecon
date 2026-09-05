"""W00 evidence manifests for archived R9 tables and the current operators."""

from __future__ import annotations

from pathlib import Path

from planetrecon import constants as C
from planetrecon.provenance import experiment_manifest, load_manifest, write_manifest


def r9_historical_manifest() -> dict:
    """Load immutable historical identities, never stamp them with current code."""
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "results/manifests/r9-historical.json")
    from planetrecon.provenance import sha256_bytes

    for name, expected in manifest["artifact_hashes"].items():
        path = root / name
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"archived R9 artifact changed or missing: {name}")
    return manifest


def w00_w01_manifest() -> dict:
    return experiment_manifest(
        "w00-w01-operator-audit",
        status="diagnostic",
        protocol=(
            "W00 versioned evidence and test tiers; W01 Dykstra positivity/support "
            "projection, forward/adjoint audit, noise-approximation quantification, "
            "and method-certificate fingerprints. CPU-only; default thread cap 8. No Q3 and no "
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
            {
                "path": "tests/test_review_w01.py",
                "status": "diagnostic",
                "kind": "regression",
            },
        ],
        notes=(
            f"Estimator operator version is {C.ESTIMATOR_OPERATOR_VERSION} (corrected Dykstra). "
            "Simulator operator version remains 1.0. Historical R9 result files are unchanged. "
            "The bounded audit is not full scientific requalification; W02/W03 and "
            "development-family ranking/gap checks remain outstanding."
        ),
        extra={
            "archived": False,
            "default_cpu_threads": C.DEFAULT_CPU_THREADS,
            "device": "cpu",
        },
    )


def w04_w08_manifest() -> dict:
    return experiment_manifest(
        "w04-w08-vertical-slice",
        status="diagnostic",
        protocol=(
            "W04 FrameSource/config/result contracts; W05 owned jobs and "
            "atomic snapshots; W06 SER v3 reader; W07 baseline stack and raw CFA RGB backprojection; "
            "W08 Qt6 raster shell and Linux PyInstaller spec. CPU default 32 threads. "
            "GPU Auto/GPU paths probe live operators and fall back when the torch "
            "build lacks the device architecture (Debian torch 2.6 / sm_120)."
        ),
        seed_coverage={"unit_fixtures": True, "scientific_families": False},
        results=[
            {"path": "tests/test_w04_w08.py", "status": "diagnostic", "kind": "unit"},
            {"path": "tests/test_review_w04_w08.py", "status": "diagnostic", "kind": "regression"},
        ],
        notes=(
            "Does not regenerate R9 tables or start Q3. Advanced MFBD remains gated "
            "by W03. GPU kernels are not claimed on unsupported architectures. "
            "Coverage, non-finite rejection, SER validation, calibrated units, worker "
            "cancellation and Qt window-close regressions are CPU fixtures. This is a "
            "prototype: iterative CFA inversion, checkpoint resume, hard memory budgets, "
            "parent-crash recovery and clean-system release acceptance remain outstanding."
        ),
        extra={
            "archived": False,
            "default_cpu_threads": C.DEFAULT_CPU_THREADS,
            "device": "auto",
            "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
            "ser_operator_version": C.SER_OPERATOR_VERSION,
            "job_schema_version": C.JOB_SCHEMA_VERSION,
        },
    )


def w09_manifest() -> dict:
    return experiment_manifest(
        "w09-field-surface-geometry",
        status="diagnostic",
        protocol=(
            "W09 field rotation and rigid oblate-globe surface operators with tested "
            "bilinear adjoints, pose unwrapping, degeneracy reporting, and geometry-aware "
            "baseline accumulation. CFA stays in detector coordinates. CPU fixtures only. "
            "No Q3, no R9 family rerun, no Saturn layers (W10)."
        ),
        seed_coverage={"unit_fixtures": True, "scientific_families": False},
        results=[
            {"path": "tests/test_w09.py", "status": "diagnostic", "kind": "unit"},
            {"path": "tests/test_review_w09.py", "status": "diagnostic", "kind": "regression"},
        ],
        notes=(
            "Geometry-only reconstruction. Does not regenerate R9 tables or start Q3. "
            "W02/W03 remain the scientific repair path. Saturn globe/ring layers are W10. "
            "MFBD with geometry remains gated by W03/W11. Freeze-mid-exposure is the default "
            "and reports limb motion during T_exp rather than claiming a full time quadrature."
        ),
        extra={
            "archived": False,
            "default_cpu_threads": C.DEFAULT_CPU_THREADS,
            "device": "cpu",
            "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
            "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
        },
    )


def w10_manifest() -> dict:
    return experiment_manifest(
        "w10-saturn-layers",
        status="diagnostic",
        protocol=(
            "W10 Saturn globe/ring layers with near/far occlusion, static short-clip "
            "rings, illumination/shadow masks, moving-moon masking, ring-aware disc "
            "fit and region-separated coverage. CPU fixtures. No Q3 and no R9 family rerun."
        ),
        seed_coverage={"unit_fixtures": True, "scientific_families": False},
        results=[
            {"path": "tests/test_w10.py", "status": "diagnostic", "kind": "unit"},
            {"path": "tests/test_review_w10.py", "status": "diagnostic", "kind": "regression"},
        ],
        notes=(
            "Rings are a static axisymmetric profile plus optional azimuthal clumps; "
            "they do not inherit globe spin. Edge-on projected bands and mixed transparent "
            "foreground-ring/globe samples are masked. Physical radii and signed opening "
            "are required; the automatic fit is diagnostic. Shadow attenuation is illustrative. "
            "Does not regenerate R9 tables. W02/W03 remain the scientific path; W11 MFBD "
            "with geometry stays gated by W03."
        ),
        extra={
            "archived": False,
            "default_cpu_threads": C.DEFAULT_CPU_THREADS,
            "device": "cpu",
            "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
            "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
        },
    )


def write_evidence_manifests(results_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    dest = results_dir / "manifests"
    dest.mkdir(parents=True, exist_ok=True)
    written = [
        write_manifest(dest / "r9-historical.json", r9_historical_manifest()),
        write_manifest(dest / "w00-w01.json", w00_w01_manifest()),
        write_manifest(dest / "w04-w08.json", w04_w08_manifest()),
        write_manifest(dest / "w09.json", w09_manifest()),
        write_manifest(dest / "w10.json", w10_manifest()),
    ]
    for path in written:
        load_manifest(path)
    return written
