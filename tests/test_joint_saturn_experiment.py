import json

import numpy as np
import pytest

from tools.joint_saturn_experiment import combine


def write_shard(path, positions, values, metadata_positions=None):
    shape = (2, 2, 3, 4)
    signal, support = np.zeros(shape), np.zeros(shape)
    for position, value in zip(positions, values):
        signal[:, position % 2] += value
        support[:, position % 2] += 1
    metadata = dict(hashes={'model': 'fixed'}, config={'halves': 2},
        fit_statistics=[dict(position=p) for p in (positions if metadata_positions is None else metadata_positions)])
    np.savez(path, positions=np.array(positions, dtype=int), signal=signal, support=support,
             metadata=json.dumps(metadata))


def test_half_stacks_preserve_weighted_sums_across_shards(tmp_path):
    write_shard(tmp_path/'shard_0.npz', [0, 1], [2., 8.])
    write_shard(tmp_path/'shard_2.npz', [2, 3], [4., 12.])
    total, support, stats = combine(tmp_path, 4, (2, 2, 3, 4), {'model': 'fixed'}, {'halves': 2})
    np.testing.assert_array_equal(total[:, 0]/support[:, 0], 3.)
    np.testing.assert_array_equal(total[:, 1]/support[:, 1], 10.)
    np.testing.assert_array_equal(total.sum(1)/support.sum(1), 6.5)
    assert [s['position'] for s in stats] == [0, 1, 2, 3]


def test_misassociated_frame_diagnostics_are_rejected(tmp_path):
    write_shard(tmp_path/'shard_0.npz', [0, 1], [2., 8.], [1, 0])
    with pytest.raises(ValueError, match='diagnostics'):
        combine(tmp_path, 2, (2, 2, 3, 4), {'model': 'fixed'}, {'halves': 2})
