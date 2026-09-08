import hashlib
import json
from pathlib import Path


def test_archived_scene_model_evidence_complete_and_immutable():
    base = Path(__file__).resolve().parents[1]/'results'
    for folder, frames in (('p1-scene-detector-pilot', 10), ('p1-scene-detector-full', 500)):
        root = base/folder
        for name, expected in json.loads((root/'checksums.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
        report = json.loads((root/'report.json').read_text())
        protocol = json.loads((root/'protocol.json').read_text())
        assert report['status'] == 'valid' and report['complete'] and report['source_unchanged']
        assert report['q3_authorized'] is False
        assert {(r['seed'], r['dr0']) for r in report['cases']} == {(s, d) for s in (1001, 1002, 1003) for d in (4., 8.)}
        for row in report['cases']:
            assert row['input_unchanged']
            assert len(set(row['indices'])) == frames and min(row['indices']) == 0 and max(row['indices']) == 499
            assert {c['crop'] for c in row['crops']} == {'feature', 'bland'}
            for c in row['crops']:
                assert c['numerical_consistency_passed']
                assert c['models']['extended']['per_frame_mean_standardized_square'] <= protocol['per_frame_tolerance']
                assert c['models']['extended']['stack_max_standardized_square'] <= protocol['stack_max_tolerance']
