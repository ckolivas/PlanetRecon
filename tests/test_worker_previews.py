"""GUI workers need snapshots/progress, not the unused preview stream."""
import pickle
import threading
import numpy as np
import pytest
from planetrecon.io.ser import write_ser
from planetrecon.jobs import _worker_run, load_checkpoint
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult
from test_w14 import gui, pump


def capture(tmp_path):
    y, x = np.indices((48, 64))
    frame = (1000*np.exp(-((x-32)**2+(y-24)**2)/150)*(1+.1*np.cos(x))).astype('u2')
    return write_ser(tmp_path/'input.ser', np.repeat(frame[None], 8, axis=0), color_id=8)


@pytest.mark.parametrize('cancel', [False, True])
def test_disabling_preview_keeps_snapshots_progress_checkpoints_and_terminal(tmp_path, monkeypatch, cancel):
    path = capture(tmp_path)
    cfg = ReconstructionConfig(frame_preselection=False, device='cpu', threads=2, batch_frames=2)
    all_events = []
    for enabled in (True, False):
        events = []
        request = threading.Event()
        request.set()
        cancelled = threading.Event()
        class Queue:
            def put(self, event, **kwargs):
                # Match process transport ownership, including full-resolution arrays.
                events.append(pickle.loads(pickle.dumps(event, protocol=5)))
                if event.kind == 'snapshot':
                    request.set()
                    if cancel:
                        cancelled.set()
            put_nowait = put
            def cancel_join_thread(self):
                pass
        def unexpected_preview(*args, **kwargs):
            pytest.fail('An unsubscribed preview must not even be constructed')
        with monkeypatch.context() as patch:
            if not enabled:
                patch.setattr(ReconstructionResult, 'copy_preview', unexpected_preview)
            _worker_run(str(enabled), str(path), cfg.to_dict(), Queue(), cancelled,
                        str(tmp_path), request, emit_previews=enabled)
        assert any(e.kind == 'preview' for e in events) == enabled
        assert any(e.kind == 'snapshot' for e in events)
        assert any(e.kind == 'progress' for e in events)
        assert events[-1].kind == ('cancelled' if cancel else 'completed')
        assert not any(e.kind == 'error' for e in events)
        checkpoint = load_checkpoint(tmp_path/f'{enabled}.npz', cfg)
        snapshot = [e.payload for e in events if e.kind == 'snapshot'][-1]
        for name in ('image', 'coverage', 'validity'):
            np.testing.assert_array_equal(checkpoint[name], snapshot[name])
        all_events.append([e for e in events if e.kind != 'preview'])
    assert [e.kind for e in all_events[0]] == [e.kind for e in all_events[1]]
    for before, after in zip(*all_events):
        if before.kind in ('snapshot', 'completed'):
            for name in ('image', 'coverage', 'validity'):
                np.testing.assert_array_equal(before.payload[name], after.payload[name])


def test_gui_subscribes_to_full_snapshots_without_previews(gui, tmp_path, monkeypatch):
    import planetrecon.gui.app as app_module
    app, win = gui
    win.path = capture(tmp_path)
    calls = []
    start = app_module.start_stack_job
    def recorded(*args, **kwargs):
        calls.append(kwargs)
        return start(*args, **kwargs)
    monkeypatch.setattr(app_module, 'start_stack_job', recorded)
    win._start(inspect_only=False)
    pump(app, lambda: win.job is None)
    assert calls[0]['emit_previews'] is False
    assert not win.error.text()
    assert win.last_result is not None and not win.last_result.incomplete
