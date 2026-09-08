import json
from pathlib import Path

import pytest

from planetrecon.validate import check_padding_and_grid


def test_full_grid_control_coverage_and_tolerances():
    report = json.loads((Path(__file__).resolve().parents[1]/
                         'results/r10-full-grid-physics-audit/report.json').read_text())
    assert report['all_controls_passed'] and not report['q3_authorized']
    assert {(r['seed'], r['dr0'], r['crop']) for r in report['cases']} == {
        (s, d, c) for s in (1001, 1002, 1003) for d in (4., 8.) for c in ('feature', 'bland')}
    assert sum(len(r['checks']) for r in report['cases']) == 60
    for row in report['cases']:
        assert all(c['passed'] for c in row['checks'])
        assert {'padding_doubling', 'grid_doubling', 'exposure_quadrature_doubling',
                'tilt_grid_doubling'} <= {c['name'] for c in row['checks']}


def test_unknown_physics_crop_rejected_before_simulation():
    with pytest.raises(ValueError):
        check_padding_and_grid(crop_name='unknown')
