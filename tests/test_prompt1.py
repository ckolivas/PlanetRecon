import shutil

import h5py
import numpy as np
import pytest

from planetrecon.validate import (
    check_flux,
    check_high_band_ovals,
    check_noll,
    check_object_geometry,
    check_optics,
    check_screen_and_exposure,
    check_timing,
    run_fast_suite,
)


def _assert_all(checks):
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n".join(f"{c.name}: {c.message}" for c in failed)


def test_noll():
    _assert_all(check_noll())


def test_timing():
    _assert_all(check_timing())


def test_optics():
    _assert_all(check_optics())


def test_object_geometry():
    _assert_all(check_object_geometry())


def test_high_band_ovals():
    _assert_all(check_high_band_ovals())


def test_flux():
    _assert_all(check_flux())


def test_screen_and_exposure():
    _assert_all(check_screen_and_exposure())


def test_fast_suite():
    _assert_all(run_fast_suite())


@pytest.mark.slow
def test_lowfreq_and_grid():
    from planetrecon.validate import run_convergence_suite

    _assert_all(run_convergence_suite())


def test_generate_schema(tmp_path):
    from planetrecon import constants as C
    from planetrecon.hdf5io import filename
    from planetrecon.simulate import generate_one
    from planetrecon.validate import (
        certify_development_validations,
        certify_development_structure_function,
        check_schema_file,
    )

    path = generate_one(
        1001, 8.0, tmp_path, n_frames=2, exposure_samples_j=4
    )
    _assert_all(check_schema_file(path))
    assert path.with_suffix(".validation.txt").exists()
    with h5py.File(path, "r") as f:
        validation = f["validation"]
        for name in (
            "grid_convergence",
            "exposure_convergence",
            "padding_convergence",
            "lowfreq_convergence",
        ):
            assert np.all(np.isfinite(validation[name][...])), name
        assert not bool(validation["gate_eligible"][()])
        for oval in ("oval1", "oval2", "oval3"):
            group = f[f"object/features/{oval}"]
            assert group["feature_crop_aperture_mask"].shape == (128, 128)
            assert group["feature_crop_annulus_mask"].shape == (128, 128)
    assert "Gate-eligible: NO" in path.with_suffix(".validation.txt").read_text()

    paths = []
    for seed in C.DEV_SEEDS:
        for dr0 in C.MANDATORY_DR0:
            target = tmp_path / filename(seed, dr0)
            if target != path:
                shutil.copy2(path, target)
                with h5py.File(target, "r+") as f:
                    f.attrs["seed"] = seed
                    f["config"].attrs["Dr0"] = dr0
            paths.append(target)
    assert certify_development_structure_function(paths).passed
    for target in paths:
        _assert_all(check_schema_file(target, require_gate_eligible=True))
        assert "Gate-eligible: YES" in target.with_suffix(
            ".validation.txt"
        ).read_text()

    evaluation_target = tmp_path / filename(2010, 8.0)
    shutil.copy2(paths[0], evaluation_target)
    with h5py.File(evaluation_target, "r+") as f:
        f.attrs["seed"] = 2010
        validation = f["validation"]
        validation["lowfreq_convergence_local_pass"][...] = False
        validation["lowfreq_convergence_pass"][...] = False
        validation["gate_eligible"][...] = False
    assert certify_development_validations(paths, [evaluation_target]).passed
    with h5py.File(evaluation_target, "r") as f:
        validation = f["validation"]
        assert not bool(validation["lowfreq_convergence_local_pass"][()])
        assert bool(validation["lowfreq_convergence_pass"][()])
        assert bool(validation["gate_eligible"][()])
