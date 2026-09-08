import h5py
import numpy as np
import pytest
from tools.scene_selection_manifest import selection_rows, read_case
from tools.profile_scene_full_count import frame_indices


def test_nested_fractions_and_sum_prior_with_deterministic_ties():
    _, rows = selection_rows(np.ones((500, 6, 6)), 2.)
    previous = set()
    for row, count in zip(rows, (25, 50, 125, 250, 500)):
        assert row['indices'] == list(range(500-count, 500))
        assert previous <= set(row['indices'])
        previous = set(row['indices'])
        assert row['mean_ridge'] * count == pytest.approx(.0003)
        assert row['observed_electron_sum'] == count*36


def test_ranking_reads_no_truth_or_optical_datasets(tmp_path):
    path = tmp_path/'only-observations.h5'
    images = np.ones((500, 6, 6))
    images[0, 3, 3] = 10
    with h5py.File(path, 'w') as f:
        f['frames/feature_observed_e'] = images
        f.create_group('config').attrs['read_noise_e'] = 2.
    scores, rows = read_case(path, 'feature')
    assert scores[0] > max(scores[1:])
    assert 0 in rows[0]['indices']


@pytest.mark.parametrize('count', [11, 100, 500])
def test_profile_counts_are_unique_and_include_endpoints(count):
    indices = frame_indices(count)
    assert len(set(indices)) == count
    assert indices[0] == 0 and indices[-1] == 499


def test_nonfinite_ranking_cannot_be_selected():
    images = np.ones((500, 6, 6)); images[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        selection_rows(images, 2.)
