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
