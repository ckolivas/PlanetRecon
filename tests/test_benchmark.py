import json
import numpy as np
from planetrecon.benchmark import run, split_half_metric
from planetrecon.io.ser import write_ser, COLOR_RGGB


def test_private_bayer_benchmark_records_scope(tmp_path):
    yy,xx=np.indices((12,16))
    frame=(10+100*np.exp(-((yy-6)**2+(xx-8)**2)/12)).astype('u1')
    path=write_ser(tmp_path/'private-observer.ser',np.stack([frame]*6),color_id=COLOR_RGGB)
    report=run(path,tmp_path/'bench','fixture',sample_frames=6)
    assert report['full_capture'] and report['used_frames']==6
    assert report['split_half']['relative_rms']==0
    assert report['raw_residual']['relative_rms'] < 1e-14
    assert not report['distribution_permission'] and not report['geometry_qualified']
    assert str(path) not in json.dumps(report) and 'observer' not in json.dumps(report)
    assert (tmp_path/'bench/result.npz').exists()


def test_mono_override_and_uniform_whole_capture_sampling(tmp_path):
    frame=np.arange(192,dtype='u2').reshape(12,16)
    path=write_ser(tmp_path/'infrared.ser',np.stack([frame]*10),color_id=COLOR_RGGB)
    report=run(path,tmp_path/'mono','mono-fixture',sample_frames=4,max_frames=4,color_override='mono',uniform=True)
    assert report['source_frames']==10 and report['processed_frames']==4
    assert report['color']=='mono' and report['color_override']=='mono'
    assert report['sampling']=='uniform capture subsample' and not report['full_capture']
    assert report['split_half']['relative_rms']==0
