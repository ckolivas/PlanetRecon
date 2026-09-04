import json
from pathlib import Path

import pytest

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.provenance import (
    RESULT_STATUSES,
    certificate_is_current,
    config_hash,
    current_method_fingerprint,
    experiment_manifest,
    fingerprint_mismatch,
    fingerprints_compatible,
    load_manifest,
    manifest_errors,
    physics_compatible,
    source_hash,
    validate_status,
    write_manifest,
)


def test_source_and_config_hashes_are_stable():
    a = source_hash()
    b = source_hash()
    assert a == b
    assert len(a) == 64
    cfg = make_config(1001, 8.0)
    assert config_hash(cfg) == config_hash(cfg)
    assert config_hash() == config_hash()


def test_result_statuses_are_the_declared_set():
    assert RESULT_STATUSES == (
        "diagnostic",
        "valid",
        "incomplete",
        "invalid",
        "gate_passed",
    )
    for status in RESULT_STATUSES:
        assert validate_status(status) == status
    with pytest.raises(ValueError):
        validate_status("passed")


def test_manifest_roundtrip(tmp_path):
    manifest = experiment_manifest(
        "w00-cpu-fixture",
        status="diagnostic",
        protocol="W00 provenance unit test",
        seed_coverage={"dev": list(C.DEV_SEEDS)},
        tolerances={"dykstra": C.DYKSTRA_TOL},
        results=[{"path": "results/q2/closure_eval_feature.json", "status": "valid"}],
        notes="does not modify archived R9 tables",
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_manifest(path)
    assert loaded["experiment_id"] == "w00-cpu-fixture"
    assert loaded["status"] == "diagnostic"
    assert loaded["source_hash"] == source_hash()
    assert not manifest_errors(loaded)


def test_manifest_rejects_unknown_status(tmp_path):
    manifest = experiment_manifest(
        "bad",
        status="diagnostic",
        protocol="x",
    )
    manifest["status"] = "pretty"
    with pytest.raises(ValueError, match="invalid manifest"):
        write_manifest(tmp_path / "bad.json", manifest)


def test_historical_r9_tables_are_archived_unchanged():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "results/prompt2/classification_eval_feature.json",
        root / "results/prompt2/classification_eval_bland.json",
        root / "results/q2/closure_eval_feature.json",
        root / "results/q2/closure_eval_bland.json",
        root / "results/q2/holdout_dev_seed-01001_feature.json",
        root / "results/manifests/r9-historical.json",
    ]
    for path in required:
        assert path.is_file(), path
    q2 = json.loads((root / "results/q2/closure_eval_feature.json").read_text())
    assert q2["4.0"]["median_C"] == pytest.approx(0.36388457737905266)
    archived = json.loads((root / "results/manifests/r9-historical.json").read_text())
    assert archived["status"] in RESULT_STATUSES
    assert archived["experiment_id"] == "r9-historical-bundle"
    assert archived["archived"] is True


def test_fingerprint_physics_ignores_seed_and_frames():
    a = current_method_fingerprint(make_config(1001, 8.0, n_frames=2, exposure_samples_j=4))
    b = current_method_fingerprint(make_config(2010, 4.0, n_frames=500, exposure_samples_j=4))
    assert physics_compatible(a, b)
    c = current_method_fingerprint(make_config(1001, 8.0, n_frames=2, exposure_samples_j=8))
    assert not physics_compatible(a, c)
    assert any("exposure_samples_J" in item for item in fingerprint_mismatch(a, c))


def test_stale_operator_version_is_not_current():
    current = current_method_fingerprint()
    stale = dict(current)
    stale["simulator_operator_version"] = "0.9"
    assert fingerprints_compatible(current, current)
    assert not certificate_is_current(stale, current)
