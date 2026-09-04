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
    from planetrecon.simulate import generate_one
    from planetrecon.validate import check_schema_file

    path = generate_one(
        1001, 8.0, tmp_path, n_frames=2, exposure_samples_j=4
    )
    _assert_all(check_schema_file(path))
    assert path.with_suffix(".validation.txt").exists()
