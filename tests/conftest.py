import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planetrecon.runtime import apply_thread_limits

# Default suite is CPU-backed; device-probe regressions use simulated devices.
apply_thread_limits()


@pytest.fixture(autouse=True)
def isolated_preferences(tmp_path_factory, monkeypatch):
    """GUI windows save preferences; keep them out of the user's own files."""
    directory = tmp_path_factory.mktemp('preferences')
    monkeypatch.setenv('PLANETRECON_SETTINGS_PATH', str(directory/'settings.json'))
    monkeypatch.setenv('PLANETRECON_SER_SETTINGS_PATH', str(directory/'ser-settings.json'))


@pytest.fixture(autouse=True)
def float64_registration(request, monkeypatch):
    """Device stacks are compared with the CPU reference at float64.

    Production registration runs in float32 (backends.precision); tests
    marked ``registration_precision`` keep that setting.
    """
    if request.node.get_closest_marker('registration_precision') is not None:
        return
    try:
        import torch
        import planetrecon.backends.precision as precision
    except ImportError:
        return
    monkeypatch.setattr(precision, 'REGISTRATION', torch.float64)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run full development-seed simulations and convergence checks",
    )
    parser.addoption(
        "--run-scientific",
        action="store_true",
        default=False,
        help="run expensive scientific seed families",
    )
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="run GPU or release-hardware tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "scientific: expensive scientific seed families (not in the default suite)"
    )
    config.addinivalue_line(
        "markers", "hardware: GPU or release-hardware tests (not in the default suite)"
    )
    config.addinivalue_line(
        "markers", "registration_precision: keep production float32 device registration"
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="slow tests are opt-in; pass --run-slow")
    skip_sci = pytest.mark.skip(reason="scientific families are opt-in; pass --run-scientific")
    skip_hw = pytest.mark.skip(reason="hardware tests are opt-in; pass --run-hardware")
    run_slow = config.getoption("--run-slow")
    run_sci = config.getoption("--run-scientific")
    run_hw = config.getoption("--run-hardware")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "scientific" in item.keywords and not run_sci:
            item.add_marker(skip_sci)
        if "hardware" in item.keywords and not run_hw:
            item.add_marker(skip_hw)
