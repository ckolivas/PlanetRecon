from dataclasses import replace

import numpy as np
import pytest

from planetrecon.io.ser import SERSource, write_ser, header_exposure_s
from planetrecon.geometry.pose import capture_exposure
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import prepare_geometry
from planetrecon.pipeline.preprocess_cache import preprocess_source, load_cache
from planetrecon.reconstruction import ReconstructionConfig


@pytest.mark.parametrize('text,seconds', [
    ('fps=50.13gain=247exp=20.00', .020),
    ('fps=327.87gain=313exp=3.00', .003),
    ('fps=76.92gain=331exp=13.00', .013),
    (' fps=2 gain=10 exp=500 ', .5),
    ('fps=10gain=1exp=2e1', .020),
])
def test_recorded_exposure_is_milliseconds(text, seconds):
    assert header_exposure_s(text) == pytest.approx(seconds)


@pytest.mark.parametrize('text', ['', 'Celestron 11', 'exp=20', 'fps=50gain=1exp=nan',
    'fps=50gain=1exp=1e999', 'fps=0gain=1exp=20', 'fps=50gain=1exp=-20',
    'fps=50gain=1exp=0', 'fps=50gain=1exp=20ms', 'fps=50gain=1exp=20 junk'])
def test_unknown_or_malformed_header_does_not_invent_exposure(text):
    assert header_exposure_s(text) is None


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_rate_rad_s=.5, field_center_x=24.,
        field_center_y=20., equatorial_radius_px=12., **kwargs)


def capture(tmp_path, telescope='fps=50gain=100exp=20.00'):
    y, x = np.indices((40, 48))
    image = (((x-24)**2+(y-20)**2 < 12**2) * (3000 + 200*np.cos(y))).astype('u2')
    frames = np.repeat(image[None], 16, axis=0)
    # Deliberately different from the 20ms exposure: never infer from cadence.
    ticks = 638_900_000_000_000_000 + np.arange(16, dtype=np.int64)*500_000
    return write_ser(tmp_path/'timed.ser', frames, timestamps=ticks, telescope=telescope)


def test_auto_exposure_changes_pose_and_matches_explicit_run(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        assert source.metadata().extras['exposure_s'].origin == 'header'
        cfg = config()
        assert cfg.exposure_s is None
        poses, _, _, _ = prepare_geometry(source, cfg)
        assert poses[0].t_s == .01
        assert poses[0].field_angle_rad == .005
        assert poses[1].t_s == pytest.approx(.06)
        automatic = stack_source(source, cfg)
        explicit = stack_source(source, replace(cfg, exposure_s=.020))
        np.testing.assert_array_equal(automatic.image, explicit.image)
        np.testing.assert_array_equal(automatic.coverage, explicit.coverage)
        assert automatic.provenance['exposure']['origin'] == 'header'
        assert automatic.provenance['config']['exposure_s'] is None
        zero, _, _, _ = prepare_geometry(source, replace(cfg, exposure_s=0.))
        assert zero[0].t_s == 0.
        manual, _, _, _ = prepare_geometry(source, replace(cfg, exposure_s=.010))
        assert manual[0].t_s == .005


def test_preprocessing_reports_exposure_and_refreshes_geometry_only(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = replace(config(), frame_preselection=True)
        selection = preprocess_source(source, cfg)
        assert selection.summary['geometry_estimate']['exposure']['value_s'] == .020
        source.header = replace(source.header, telescope='fps=50gain=100exp=10.00')
        cached, report = load_cache(source, cfg)
        assert cached is not None  # The pixel measurements remain reusable.
        np.testing.assert_array_equal(cached.accepted, selection.accepted)
        assert report['geometry_estimate']['applicable'] is False
        assert 'Exposure changed' in report['geometry_estimate']['notes'][0]


def test_missing_header_uses_no_offset_and_manual_zero_is_preserved(tmp_path):
    with SERSource(capture(tmp_path, telescope='My telescope')) as source:
        assert capture_exposure(source)['origin'] == 'unavailable'
        assert capture_exposure(source)['value_s'] == 0.
        assert capture_exposure(source, 0.) == {'value_s': 0., 'origin': 'user'}
        poses, _, _, _ = prepare_geometry(source, config())
        assert poses[0].t_s == 0.


def test_auto_exposure_cannot_bypass_unsupported_quadrature_check(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        with pytest.raises(ValueError, match='quadrature'):
            prepare_geometry(source, config(freeze_mid_exposure=False))


def test_gui_auto_exposure_tracks_capture_without_overwriting_manual_value(tmp_path):
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app([])
    win = MainWindow()
    try:
        with SERSource(capture(tmp_path)) as source:
            def inspect():
                win._set_input({'source_metadata': source.metadata().as_dict(),
                                'input_image': source.read_raw(0), 'input_view': 'mono'})
            inspect()
            edit = win.controls.fields['exposure_s']
            assert win.controls.configuration().exposure_s is None
            assert edit.placeholderText() == 'Automatic: 0.02 s'
            edit.setText('0.004')
            inspect()
            assert win.controls.configuration().exposure_s == .004
            edit.clear()
            source.header = replace(source.header, telescope='fps=50gain=100exp=3')
            inspect()
            assert edit.placeholderText() == 'Automatic: 0.003 s'
            assert win.controls.configuration().exposure_s is None
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()


def test_auto_exposure_resume_preserves_result_and_rejects_changed_header(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = config(batch_frames=2)
        whole = stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            if info['n_processed'] >= 4:
                stop = True
        checkpoint = tmp_path/'resume.npz'
        partial = stack_source(source, cfg, should_cancel=lambda: stop,
                               on_event=event, state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        source.header = replace(source.header, telescope='fps=50gain=100exp=10.00')
        with pytest.raises(ValueError, match='identity|mismatch|configuration'):
            stack_source(source, cfg, resume_from=checkpoint)
