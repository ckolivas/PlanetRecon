import numpy as np
from tools.validate_pooled_reference import unused_positions


def test_unused_selection_and_half_slots_are_complete_disjoint_and_balanced():
    indices=np.arange(5738)*4
    previous=np.linspace(0,5737,512,dtype=int)
    native=np.arange(100,356).reshape(4,64)
    historical=indices[500:564]
    selected=unused_positions(indices,previous,native,historical)
    assert len(np.unique(selected))==1024
    assert not np.intersect1d(selected,previous).size
    assert not np.intersect1d(selected,native).size
    assert not np.intersect1d(indices[selected],historical).size
    for cohort in (0,1):
        ids=indices[selected[cohort::2]]
        assert len(ids)==512
        for half in (0,1):
            positions=np.arange(1024)[np.arange(1024)%4==cohort+2*half]
            np.testing.assert_array_equal(indices[selected[positions]],ids[half::2])
            assert len(positions)==256
