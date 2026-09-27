import json
import numpy as np
import pytest

from tools.coherent_saturn_experiment import combine_shards


def shard(path,positions,signal,support,hashes=None):
    metadata = {'hashes':{'reference':'fixed'} if hashes is None else hashes,
                'config':{'spacing':16},'fit_statistics':[{} for _ in positions]}
    np.savez(path,positions=positions,signal=np.full((2,3),signal,dtype=float),
             support=np.full((2,3),support,dtype=float),metadata=json.dumps(metadata))
    return path


def test_shards_combine_weighted_sums_before_normalizing(tmp_path):
    a=shard(tmp_path/'a.npz',[0,1],10.,2.)
    b=shard(tmp_path/'b.npz',[2],30.,3.)
    signal,support,stats=combine_shards([a,b],3,(2,3),{'reference':'fixed'},{'spacing':16})
    np.testing.assert_array_equal(signal/support,8.)
    assert len(stats)==3


@pytest.mark.parametrize('positions',[[0,0,1],[0,1]])
def test_duplicate_or_missing_frames_cannot_be_exported(tmp_path,positions):
    a=shard(tmp_path/'a.npz',positions,10.,2.)
    with pytest.raises(ValueError,match='each selected frame once'):
        combine_shards([a],3,(2,3),{'reference':'fixed'},{'spacing':16})


def test_mixed_reference_shards_are_rejected(tmp_path):
    a=shard(tmp_path/'a.npz',[0],10.,2.,hashes={'reference':'different'})
    with pytest.raises(ValueError,match='identity mismatch'):
        combine_shards([a],1,(2,3),{'reference':'fixed'},{'spacing':16})
