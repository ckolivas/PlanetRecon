"""Preprocessing and subsequent motion runs share the capture midpoint epoch."""
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from planetrecon.jobs import _worker_run
from planetrecon.io.ser import write_ser
from planetrecon.reconstruction import ReconstructionConfig
from test_w14 import gui


def ready(duration):
    return {'status': 'ready', 'n_total': 16, 'accepted': 16, 'excluded': 0,
            'quality': 0, 'shape': 0, 'quality_shape_overlap': 0, 'other': 0,
            'timing': {'status': 'available', 'duration_s': duration},
            'geometry_estimate': {'suggestions': {}}}


def test_cache_prefills_midpoint_and_next_run_uses_it(gui, tmp_path, monkeypatch):
    _, window = gui
    configs = []
    def start(path, cfg, **options):
        configs.append(cfg)
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    window.path = tmp_path/'input.ser'
    window._set_preprocessing(ready(75.))
    assert window.controls.configuration().reference_epoch_s == 37.5
    window._run()
    assert configs[-1].reference_epoch_s == 37.5
    window._finish_job()
    window.controls.clear_geometry_estimate()
    assert window.controls.configuration().reference_epoch_s == 0.
    window._set_preprocessing(ready(360.))
    assert window.controls.configuration().reference_epoch_s == 180.


@pytest.mark.parametrize('value', [0., 12.3])
def test_manual_epoch_including_zero_is_retained(gui, value):
    _, window = gui
    edit = window.controls.fields['reference_epoch_s']
    edit.setText(str(value))
    edit.textEdited.emit(str(value))
    window._set_preprocessing(ready(100.))
    assert window.controls.configuration().reference_epoch_s == value
    assert not window.controls.wants_midpoint_epoch()


def test_loaded_nondefault_and_checkpoint_epochs_are_retained(gui):
    _, window = gui
    edit = window.controls.fields['reference_epoch_s']
    edit.setText('23')  # Loaded nondefault configuration, not a keyboard edit.
    window._set_preprocessing(ready(100.))
    assert window.controls.configuration().reference_epoch_s == 23.
    edit.setText('0')
    window.checkpoint_path.setText('/tmp/existing-state.npz')
    window._set_preprocessing(ready(100.))
    assert window.controls.configuration().reference_epoch_s == 0.


@pytest.mark.parametrize('timing', [{}, {'status': 'unavailable', 'duration_s': None},
                                  {'status': 'invalid', 'duration_s': None},
                                  {'status': 'available', 'duration_s': float('nan')}])
def test_unknown_timing_does_not_invent_midpoint(gui, timing):
    _, window = gui
    info = ready(0.)
    info['timing'] = timing
    window._set_preprocessing(info)
    assert window.controls.configuration().reference_epoch_s == 0.
    assert 'reference_epoch_s' not in window.controls.geometry_auto


@pytest.mark.parametrize('auto,epoch', [(True, 0.), (False, 0.), (False, 3.)])
def test_worker_estimates_and_reloads_geometry_at_the_chosen_epoch(tmp_path, monkeypatch, auto, epoch):
    y,x = np.indices((48,48))
    image = (1000*np.exp(-((x-24)**2+(y-24)**2)/90)*(1+.1*np.cos(x))).astype('u2')
    # Irregular timestamps with a tie: midpoint comes from endpoints, not the
    # middle frame index, selected best frame or an estimated mean cadence.
    times = np.r_[0.,0.,np.linspace(.1,1.3,13),10.]
    ticks = np.uint64(638_900_000_000_000_000)+np.rint(times*1e7).astype('u8')
    path = write_ser(tmp_path/'input.ser', np.repeat(image[None],16,axis=0), timestamps=ticks)
    cfg = ReconstructionConfig(device='cpu', threads=2, reference_epoch_s=epoch)
    seen = []
    def discover(source, config, *args):
        seen.append(config.reference_epoch_s)
        return {'status': 'unresolved', 'suggestions': {}, 'notes': []}
    monkeypatch.setattr('planetrecon.geometry.discovery.discover_geometry', discover)
    events = queue.Queue()
    _worker_run('pre', str(path), cfg.to_dict(), events, threading.Event(), None,
                preprocess_only=True, auto_output_epoch=auto)
    messages = list(events.queue)
    assert not [e.payload for e in messages if e.kind == 'error']
    expected = 5. if auto else epoch
    assert seen == [expected]
    completed = next(e.payload for e in messages if e.kind == 'completed')
    assert completed['preprocessing_cache']['timing']['duration_s'] == 10.
    events = queue.Queue()
    _worker_run('inspect', str(path), cfg.to_dict(), events, threading.Event(), None,
                inspect_only=True, auto_output_epoch=auto)
    info = next(e.payload['preprocessing_cache'] for e in list(events.queue) if e.kind == 'completed')
    assert info['status'] == 'ready'
    assert info['geometry_estimate'].get('applicable', True)


def test_preprocessing_options_respect_manual_zero_and_checkpoints(gui, tmp_path, monkeypatch):
    _, window = gui
    options = []
    def start(path, cfg, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    window.path = tmp_path/'input.ser'
    window._preprocess()
    assert options[-1]['auto_output_epoch']
    window._finish_job()
    window.controls.fields['reference_epoch_s'].textEdited.emit('0')
    window._preprocess()
    assert not options[-1]['auto_output_epoch']
    window._finish_job()
