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
