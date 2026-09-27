import json
import numpy as np
import pytest

from tools.refit_saturn_experiment import cached_validation


def fixture(tmp_path, positions=(0,1), change=None):
    hashes = {key:'fixed' for key in ('reference','indices','quality','global_shifts','model_code','validated_registration')}
    stored = {key:value for key,value in hashes.items() if key!='validated_registration'}
    stored['validation_code'] = 'fixed'
    metadata = dict(hashes=stored,config=dict(spacing=16.,stiffness=.01,patch_average=True,independent_pixel_validation=True),
                    fit_statistics=[dict(accepted_halves=int(i%3),field_guard_accepted=True) for i in positions])
    if change:
        change(metadata)
    np.savez(tmp_path/'shard_00000.npz',positions=positions,metadata=json.dumps(metadata))
    return hashes


def test_cached_decisions_are_indexed_by_frame_position(tmp_path):
    hashes = fixture(tmp_path,(1,0))
    result = cached_validation(tmp_path,hashes,2)
    assert [s['accepted_halves'] for s in result] == [0,1]


@pytest.mark.parametrize('positions',[(0,),(0,0),(0,2)])
def test_missing_duplicate_or_out_of_range_decisions_are_rejected(tmp_path,positions):
    hashes = fixture(tmp_path,positions)
    with pytest.raises(ValueError):
        cached_validation(tmp_path,hashes,2)


@pytest.mark.parametrize('change',[
    lambda m:m['hashes'].update(quality='different'),
    lambda m:m['hashes'].update(validation_code='different'),
    lambda m:m['config'].update(stiffness=.1),
    lambda m:m['fit_statistics'][0].update(accepted_halves=3),
])
def test_incompatible_decisions_are_rejected(tmp_path,change):
    hashes = fixture(tmp_path,change=change)
    with pytest.raises(ValueError):
        cached_validation(tmp_path,hashes,2)
