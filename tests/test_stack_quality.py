"""Stack quality summary: frame counts, weighting, effective frames and SNR."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import COLOR_RGGB, SERSource, write_ser
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.pipeline.stack_quality import describe, frame_noise, summarise
from planetrecon.reconstruction import ReconstructionConfig


def scene(shape=(120, 144)):
    y, x = np.indices(shape)
    disc = 300.*(np.hypot(x - shape[1]/2, y - shape[0]/2) < shape[0]/3) + 30
    return gaussian_filter(disc*(1 + .05*np.sin(x/8)*np.cos(y/10)), 2.5)


@pytest.mark.parametrize('mode', ['mono', 'RGGB'])
@pytest.mark.parametrize('sigma', [2., 8.])
def test_frame_noise_recovers_known_noise_and_signal(mode, sigma):
    truth = scene()
    frame = truth + np.random.default_rng(3).normal(0, sigma, truth.shape)
    signal, noise = frame_noise(frame, mode)
    assert noise == pytest.approx(sigma, rel=.05)
    assert signal == pytest.approx(300, rel=.1)  # Mean over the bright target; the blurred limb lowers it.


def test_effective_frames_follow_the_weights():
    quality = np.full(10, np.nan)
    quality[[1, 4, 6, 9]] = [1., 1., 1., 1.]
    equal = summarise(quality, True)
    assert equal['frames_stacked'] == 4 and equal['effective_frames'] == pytest.approx(4)
    quality[[1, 4, 6, 9]] = [1., 2., 3., 4.]
    weighted = summarise(quality, True, screened=np.r_[quality[:9], 0.], frame_snr={'snr': 10.})
    assert weighted['effective_frames'] == pytest.approx(100/30)
    assert weighted['quality']['mean'] == pytest.approx(2.5)
    assert weighted['quality']['weighted_mean'] == pytest.approx(30/10)
    assert weighted['quality']['screened_mean'] == pytest.approx(1.5)
    assert weighted['estimated_snr'] == pytest.approx(10*np.sqrt(100/30))
    assert summarise(quality, False)['effective_frames'] == pytest.approx(4)
    assert 'est. SNR' in describe(weighted) and 'linear quality' in describe(weighted)


def test_alignment_points_report_their_own_frames():
    quality = np.array([1., 2., 3., np.nan, 4.])
    # Point 0 keeps frames 0 and 1; point 1 keeps 2 and 4 (frame 3 was rejected).
    local = SimpleNamespace(row={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                            chosen=np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 1]], bool))
    summary = summarise(quality, True, local_selection=local, frame_snr={'snr': 5.})
    points = summary['alignment_points']
    assert points['frames_per_point'] == {'median': 2., 'minimum': 2., 'maximum': 2.}
    assert points['effective_frames']['minimum'] == pytest.approx(9/5)
    assert points['effective_frames']['maximum'] == pytest.approx(49/25)
    assert points['weighted_quality']['maximum'] == pytest.approx(25/7)
    assert summary['snr_gain'] == pytest.approx(np.sqrt((9/5 + 49/25)/2))
    assert describe(summary).startswith('each alignment point stacks 2 frames')


def write_capture(tmp_path, n=12, bayer=False):
    rng = np.random.default_rng(5)
    truth = scene((64, 80))
    frames = np.stack([gaussian_filter(truth, 1 + (i % 4)*.4) + rng.normal(0, 3, truth.shape) for i in range(n)])
    return write_ser(tmp_path/'capture.ser', np.clip(frames, 0, 65535).astype('u2'),
                     color_id=COLOR_RGGB if bayer else 0)


@pytest.mark.parametrize('bayer', [False, True])
def test_stack_records_quality_of_accumulated_frames(tmp_path, bayer):
    path = write_capture(tmp_path, bayer=bayer)
    cfg = ReconstructionConfig(device='cpu', threads=2, batch_frames=4, stack_percent=50,
                               frame_selection_mode='frame_count')
    with SERSource(path) as source:
        selection = preprocess_source(source, cfg)
        result = stack_source(source, cfg)
    summary = result.provenance['stack_quality']
    assert summary['frames_stacked'] == result.n_used == 6
    stacked = np.sort(selection.measurements[selection.accepted, 0])[-6:]
    assert summary['quality']['mean'] == pytest.approx(stacked.mean())
    assert summary['quality']['screened_frames'] == int(selection.accepted.sum())
    assert summary['effective_frames'] == pytest.approx(stacked.sum()**2/np.sum(stacked**2))
    assert summary['single_frame']['noise'] == pytest.approx(3, rel=.2)
    assert summary['estimated_snr'] == pytest.approx(summary['single_frame']['snr']*np.sqrt(summary['effective_frames']))
    json.dumps(summary, allow_nan=False)


def test_resumed_stack_keeps_the_whole_frame_record(tmp_path):
    path = write_capture(tmp_path, n=12)
    cfg = ReconstructionConfig(frame_preselection=False, device='cpu', threads=2, batch_frames=2)
    state = tmp_path/'state.npz'
    with SERSource(path) as source:
        whole = stack_source(source, cfg)
    cancel = False

    def event(result, info):
        nonlocal cancel
        cancel = cancel or info['n_processed'] >= 4

    with SERSource(path) as source:
        stack_source(source, cfg, on_event=event, should_cancel=lambda: cancel, state_checkpoint=state)
        continued = stack_source(source, cfg, resume_from=state)
    assert continued.provenance['stack_quality'] == whole.provenance['stack_quality']
    # A checkpoint written before the record resumes, and says the summary is unavailable.
    with np.load(state, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key not in ('metadata', 'frame_quality')}
        meta = json.loads(str(data['metadata']))
    meta['arrays'].remove('frame_quality')
    np.savez(state, metadata=json.dumps(meta), **arrays)
    with SERSource(path) as source:
        legacy = stack_source(source, cfg, resume_from=state)
    np.testing.assert_array_equal(legacy.image, whole.image)
    assert 'unavailable' in legacy.provenance['stack_quality']
