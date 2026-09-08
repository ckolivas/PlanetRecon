import hashlib
import json
from pathlib import Path


def test_all_scene_solver_pilots_preserve_complete_evidence():
    base = Path(__file__).resolve().parents[1]/'results'
    for folder, passed in (('p2-scene-noise-v1', False), ('p2-scene-noise-v2-initial', False),
                            ('p2-scene-noise-v2-qualified', True)):
        root = base/folder
        for name, checksum in json.loads((root/'checksums.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == checksum
        report = json.loads((root/'report.json').read_text())
        assert report['status'] == 'diagnostic' and report['complete'] and report['q3_authorized'] is False
        assert report['all_numerical_pilots_passed'] is passed
        assert {r['case'] for r in report['cases']} == {'expected/scalar', 'expected/spatial', 'observed/scalar', 'observed/spatial'}
        for row in report['cases']:
            case = root/row['case'].replace('/', '-')
            assert hashlib.sha256((case/'protocol.json').read_bytes()).hexdigest() == row['input_protocol_sha256']
            assert hashlib.sha256((case/'report.json').read_bytes()).hexdigest() == row['input_report_sha256']
            detail = json.loads((case/'report.json').read_text())
            protocol = json.loads((case/'protocol.json').read_text())
            assert detail['indices'] == [0, 249, 499] and detail['scene_shape'] == [1152, 1152]
            assert detail['source_input_unchanged'] and detail['q3_authorized'] is False
            assert [r['maxiter'] for r in detail['runs']] == protocol['budgets']
            for info in detail['runs']:
                if info['converged']:
                    assert info['feasible'] and info['relative_solution_error_bound'] <= protocol['tolerance']
            if detail['numerical_pilot_passed']:
                assert all(r['converged'] for r in detail['runs'])
                assert detail['image_relative_change'] <= protocol['image_stability_tolerance']
