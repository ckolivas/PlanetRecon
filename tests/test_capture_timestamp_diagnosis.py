import numpy as np
import pytest
from planetrecon.io.ser import write_ser
from tools.study_io import file_hash
from tools.diagnose_capture_timestamps import diagnose


def test_duplicate_times_do_not_imply_duplicate_images(tmp_path):
    path=tmp_path/'private.ser'
    frames=np.ones((6,4,4),dtype=np.uint8);frames[3]=2;frames[5]=3
    write_ser(path,frames,timestamps=np.array([10,10,20,20,30,30]))
    r=diagnose(path,file_hash(path),max_pairs=3)
    assert [p['pixels_identical'] for p in r['sampled_pairs']]==[True,False,False]
    assert r['sampled_identical_pairs']==1 and r['duplicate_intervals']==3
    assert not r['timing_repaired'] and not r['frames_removed']


def test_sampling_is_bounded_and_spans_duplicate_sequence(tmp_path):
    path=tmp_path/'private.ser';write_ser(path,np.ones((10,4,4),dtype=np.uint8),timestamps=np.ones(10,dtype=np.int64))
    r=diagnose(path,file_hash(path),max_pairs=2)
    assert [p['first_frame'] for p in r['sampled_pairs']]==[0,8]
    assert r['duplicate_intervals']==9 and len(r['sampled_pairs'])==2


def test_missing_times_and_changed_identity(tmp_path):
    path=tmp_path/'private.ser';write_ser(path,np.ones((3,4,4),dtype=np.uint8))
    r=diagnose(path,file_hash(path))
    assert not r['timestamps_present'] and not r['sampled_pairs']
    with pytest.raises(ValueError,match='hashed intake'): diagnose(path,'0'*64)
