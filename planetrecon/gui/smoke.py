"""Bounded GUI/worker/export smoke, also runnable from the frozen executable."""
from pathlib import Path
import json
import time
import uuid

import numpy as np
import tifffile
from PySide6.QtCore import QTimer

from planetrecon.export import parse_tiff_description
from planetrecon.gui.app import MainWindow, create_app
from planetrecon.io.ser import write_ser, COLOR_RGGB
from planetrecon.reconstruction import ReconstructionConfig


def run_smoke(directory: Path, device: str = "cpu") -> int:
    directory = Path(directory) / ('gui-' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    frame = (300 + np.arange(16*24).reshape(16,24)).astype(np.uint16)
    source = write_ser(directory/'fixture.ser', np.stack([frame]*4), color_id=COLOR_RGGB)
    app = create_app(['planetrecon-gui-smoke'])
    win = MainWindow(source, ReconstructionConfig(device=device, threads=2, batch_frames=1))
    win.checkpoint_path.setText(str(directory/'state.npz'))
    outcome = {'status': 'running', 'directory': str(directory), 'requested_device': device}
    phase = 'cancel_request'
    cancel_started = None
    started = time.monotonic()
    dest = directory/'result.tif'
    timer = QTimer()
    timer.setInterval(50)

    def finish(error=None):
        timer.stop()
        outcome['elapsed_s'] = time.monotonic()-started
        outcome['status'] = 'failed' if error else 'passed'
        if error:
            outcome['error'] = str(error)
        win.window.grab().save(str(directory/'window.png'))
        win._shutdown()
        if win.export_worker is not None:
            win.export_worker.wait()
        win.window.close()
        app.quit()

    def advance():
        nonlocal phase, cancel_started
        try:
            if time.monotonic()-started > 30:
                raise TimeoutError('GUI smoke exceeded 30 seconds')
            if phase == 'cancel_request':
                cancel_started = time.monotonic()
                win._cancel()
                phase = 'cancel_wait'
            elif phase == 'cancel_wait' and win.job is None:
                outcome['cancel_latency_s'] = time.monotonic()-cancel_started
                outcome['cancel_restart'] = True
                phase = 'stack'
                win._run()
            elif phase == 'stack' and win.job is None:
                if win.error.text() or win.last_result is None:
                    raise RuntimeError(win.error.text() or 'No result')
                if device == "gpu" and win.last_result.backend != "cuda":
                    raise RuntimeError("GPU smoke fell back instead of exercising CUDA")
                assert win.last_result.n_used == 4 and not win.last_result.incomplete
                assert win.last_result.image.shape == (16,24,3)
                phase = 'resume'
                win.resume_check.setChecked(True)
                win._run()
            elif phase == 'resume' and win.job is None:
                if win.error.text():
                    raise RuntimeError(win.error.text())
                assert win.last_result.provenance['resumed_from_frame'] == 4
                assert win.last_result.n_used == 4 and not win.last_result.incomplete
                if device == 'gpu':
                    assert win.last_result.backend == 'cuda'
                outcome['checkpoint_resume'] = True
                phase = 'save'
                win.save_result(dest)
            elif phase == 'save' and win.export_worker is None:
                with tifffile.TiffFile(dest) as tf:
                    actual = tf.pages[0].asarray()
                    meta = parse_tiff_description(tf.pages[0].description)
                    assert len(tf.pages) == 1 and not tf.pages[0].is_shaped
                with tifffile.TiffFile(dest.with_name(meta['coverage_file'])) as tf:
                    valid = tf.pages[0].asarray().astype(bool)
                if valid.ndim == 2 and actual.ndim == 3:
                    valid = np.broadcast_to(valid[..., None], actual.shape)
                mapping = meta['mapping']
                expected = (win.last_result.image - mapping['black']) / (mapping['white'] - mapping['black'])
                np.testing.assert_array_equal(actual[valid], expected[valid].astype(np.float32))
                assert np.isnan(actual[~valid]).all()
                outcome.update(n_used=4, shape=[16,24,3], encoding='tiff32', backend=win.last_result.backend,
                               invalid_samples=int((~valid).sum()))
                finish()
        except Exception as exc:
            finish(exc)

    win.show()
    win._run()
    timer.timeout.connect(advance)
    timer.start()
    app.exec()
    (directory/'smoke.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome['status'] == 'passed' else 1
