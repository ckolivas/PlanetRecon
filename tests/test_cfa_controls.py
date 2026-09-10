"""Fresh-run controls and CLI/worker delivery of optional colour interpolation."""
from dataclasses import replace
import time

import numpy as np

from test_local_alignment import capture, config
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import load_snapshot


def test_gui_option_defaults_off_and_tracks_all_run_controls():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(config(stack_percent=50))
    try:
        edit = controls.fields['local_cfa_interpolation']
        assert not edit.isChecked() and edit.isEnabled() and edit.toolTip()
        edit.setChecked(True)
        chosen = controls.configuration()
        assert chosen.local_cfa_interpolation and chosen.stack_percent == 50
        assert not controls.fields['squared_quality_weights'].isEnabled()
        assert ReconstructionConfig.from_dict(chosen.to_dict()) == chosen
        controls.setEnabled(False)
        assert controls.configuration().local_cfa_interpolation
        controls.setEnabled(True)
        for key in ('local_alignment', 'frame_preselection'):
            controls.fields[key].setChecked(False)
            assert not edit.isEnabled() and not controls.configuration().local_cfa_interpolation
            controls.fields[key].setChecked(True)
            assert edit.isEnabled() and controls.configuration().local_cfa_interpolation
        geo = controls.fields['geometry_mode']
        geo.setCurrentIndex(geo.findData('field'))
        assert not controls.configuration().local_cfa_interpolation
        geo.setCurrentIndex(geo.findData('none'))
        assert controls.configuration().local_cfa_interpolation
        edit.setChecked(False)
        controls.fields['squared_quality_weights'].setChecked(True)
        assert not edit.isEnabled()
        assert not controls.configuration().local_cfa_interpolation
    finally:
        controls.close()


def test_cli_candidate_delivers_full_final_snapshot(tmp_path, monkeypatch):
    from planetrecon.cli import main
    import planetrecon.result as result_module
    path = capture(tmp_path, 8)
    with SERSource(path) as source:
        preprocess_source(source, config())
    saved = []
    original = result_module.save_snapshot
    def record(path, result):
        saved.append((result.image.shape, result.provenance.get('local_cfa_interpolation')))
        original(path, result)
    monkeypatch.setattr(result_module, 'save_snapshot', record)
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--local-cfa-interpolation', '--checkpoint', str(tmp_path/'preview.npz'),
                 '--out', str(tmp_path/'result'), '--export', 'tiff32']) == 0
    result = load_snapshot(tmp_path/'result/stack.npz')
    assert not result.incomplete and not result.provenance['local_cfa_interpolation']['preview']
    assert all(shape == result.image.shape for shape, details in saved)
    assert load_snapshot(tmp_path/'preview.npz').provenance['local_cfa_interpolation']['preview'] is False
    import tifffile
    from planetrecon.export import parse_tiff_description
    with tifffile.TiffFile(tmp_path/'result/stack.tif') as tf:
        meta = parse_tiff_description(tf.pages[0].description)
        np.testing.assert_array_equal(tf.pages[0].asarray()[result.validity], result.image.astype(np.float32)[result.validity])
    with tifffile.TiffFile(tmp_path/'result'/meta['coverage_file']) as tf:
        descriptions = [parse_tiff_description(page.description) for page in tf.pages]
        effective = [d for d in descriptions if d.get('layer', '').startswith('iid_effective_samples_')]
        assert len(effective) == 3
        assert all(d['units'] == 'independent equal-variance sample equivalents' for d in effective)


def test_spawned_worker_returns_final_candidate_and_pending_snapshot(tmp_path):
    from planetrecon.jobs import start_stack_job, result_from_payload
    cfg = config(local_cfa_interpolation=True, stack_percent=50)
    path = capture(tmp_path, 8)
    with SERSource(path) as source:
        preprocess_source(source, cfg)
    handle = start_stack_job(path, cfg, queue_size=16)
    handle.snapshot_request.set()
    received = []
    deadline = time.monotonic()+45
    try:
        while handle.state not in ('completed', 'failed', 'cancelled') and time.monotonic() < deadline:
            received.extend(handle.poll(timeout=.1))
        assert handle.state == 'completed', [(e.kind, e.payload.get('message')) for e in received]
        final = result_from_payload(next(e.payload for e in received if e.kind == 'completed'))
        assert final.image.shape == (96, 112, 3) and np.isfinite(final.image).all()
        assert not final.incomplete and not final.provenance['local_cfa_interpolation']['preview']
        assert any(e.kind == 'progress' and e.payload['stage'] == 'local CFA final fit'
                   and e.payload['fraction'] is None for e in received)
        snapshots = [e for e in received if e.kind == 'snapshot']
        assert snapshots and all(e.payload['image'].shape == final.image.shape for e in snapshots)
    finally:
        handle.close()
