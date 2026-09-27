"""Brightness is a single output gain, never a change to frame contributions."""
from dataclasses import replace
import numpy as np
import pytest
import tifffile

from planetrecon.io.source import ArraySource
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.brightness import normalise_result
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.jobs import result_payload
from test_w14 import gui


def source(bits=8, colour='mono'):
    y, x = np.indices((48, 56))
    base = 20 + 60*np.exp(-((x-28)**2+(y-24)**2)/70)
    frames = np.stack([base, .4*base + 15*np.exp(-((x-21)**2+(y-24)**2)/8), .8*base])
    if colour == 'RGB':
        frames = frames[..., None] * [.8, 1., .5]
    elif colour == 'RGGB':
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .5
    frames = (frames*(100 if bits == 16 else 1)).astype('u2' if bits == 16 else 'u1')
    return ArraySource(frames, color_mode=colour, bit_depth=bits)


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False, batch_frames=1, **kwargs)


@pytest.mark.parametrize('bits', [8, 16])
@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB'])
@pytest.mark.parametrize('weighted', [False, True])
def test_normalised_output_is_exactly_one_gain_of_intrinsic_brightness_stack(bits, colour, weighted):
    src = source(bits, colour)
    cfg = config(quality_weighting=weighted)
    original_frames = src._frames.copy()
    raw = stack_source(src, cfg)
    snapshots = []
    normal = stack_source(src, replace(cfg, normalise_brightness=True),
                         on_event=lambda result, info: snapshots.append(result))
    target = .7*((1 << bits)-1)
    gain = target / raw.image[raw.validity].max()
    np.testing.assert_allclose(normal.image, raw.image*gain, rtol=1e-14)
    np.testing.assert_array_equal(normal.coverage, raw.coverage)
    np.testing.assert_array_equal(normal.validity, raw.validity)
    np.testing.assert_array_equal(src._frames, original_frames)
    assert normal.image[normal.validity].max() == pytest.approx(target)
    assert 'brightness_normalisation' not in raw.provenance
    for snapshot in snapshots:
        if snapshot.n_used:
            assert snapshot.image[snapshot.validity].max() == pytest.approx(target)


@pytest.mark.parametrize('mode', ['field', 'surface', 'combined', 'saturn'])
def test_geometry_models_receive_same_final_gain(tmp_path, mode):
    from test_geometry_resume import capture, config as geometry_config
    path = capture(tmp_path, 'mono', mode)
    cfg = geometry_config(mode)
    if mode in ('surface', 'combined'):
        cfg = replace(cfg, field_center_x=64., field_center_y=64., equatorial_radius_px=42.,
                      sub_obs_lat_rad=0., flattening=0.)
    with SERSource(path) as src:
        raw = stack_source(src, cfg)
        normal = stack_source(src, replace(cfg, normalise_brightness=True, normalise_percent=60))
    gain = .6*65535/raw.image[raw.validity].max()
    np.testing.assert_allclose(normal.image, raw.image*gain, rtol=1e-14)
    np.testing.assert_array_equal(normal.coverage, raw.coverage)
    for key in raw.layer_coverage:
        np.testing.assert_array_equal(normal.layer_coverage[key], raw.layer_coverage[key])


def test_resume_accumulators_remain_unscaled_and_allow_new_output_percentage(tmp_path):
    src = source()
    path = write_ser(tmp_path/'input.ser', src._frames)
    cfg = config(normalise_brightness=True)
    checkpoint = tmp_path/'state.npz'
    stop = False
    def event(result, info):
        nonlocal stop
        stop = result.n_used >= 1
    with SERSource(path) as capture:
        partial = stack_source(capture, cfg, state_checkpoint=checkpoint,
                               on_event=event, should_cancel=lambda: stop)
        assert partial.incomplete
        resumed = stack_source(capture, replace(cfg, normalise_percent=80), resume_from=checkpoint)
        raw = stack_source(capture, replace(cfg, normalise_brightness=False))
    gain = .8*255/raw.image[raw.validity].max()
    np.testing.assert_allclose(resumed.image, raw.image*gain, rtol=1e-14)
    np.testing.assert_array_equal(resumed.coverage, raw.coverage)


def test_gain_calibration_uses_corresponding_electron_full_scale():
    result = stack_source(source(), config(normalise_brightness=True, gain_e_per_adu=2.5))
    assert result.units == 'e-'
    assert result.image[result.validity].max() == pytest.approx(.7*255*2.5)


def test_unsupported_peak_is_ignored_and_zero_result_is_not_invented():
    from test_w14 import result
    r = result(value=0, incomplete=False)
    r.image[0,0] = 1e9
    r.validity[0,0] = False
    normalise_result(r, 70, 255.)
    assert not r.provenance['brightness_normalisation']['applied']
    assert not r.image[r.validity].any()
    r.image[1,0] = [20, 40, 10]
    normalise_result(r, 70, 255.)
    np.testing.assert_allclose(r.image[1,0], np.array([20,40,10]) * (178.5/40))


@pytest.mark.parametrize('value', [0, 101, 70.5, True, float('nan')])
def test_invalid_percentage_is_rejected(value):
    with pytest.raises(ValueError, match='normalise_percent'):
        config(normalise_percent=value)


def test_unknown_absolute_range_is_refused():
    src = ArraySource(np.ones((2, 20, 20)), bit_depth=32)
    with pytest.raises(ValueError, match='known integer ADU range'):
        stack_source(src, config(normalise_brightness=True))


def test_gui_toggle_defaults_and_levels_preserve_percentage_in_export(gui, tmp_path):
    from planetrecon.export import export_result
    _, win = gui
    c = win.controls
    assert not c.fields['normalise_brightness'].isChecked()
    assert c.fields['normalise_percent'].value() == 70
    assert not c.fields['normalise_percent'].isEnabled()
    c.fields['normalise_brightness'].setChecked(True)
    c.fields['normalise_percent'].setValue(50)
    assert c.fields['normalise_percent'].isEnabled()
    c.restore_settings(c.settings_state())
    assert c.fields['normalise_percent'].isEnabled()
    output = stack_source(source(), config(normalise_brightness=True, normalise_percent=50))
    win.auto_levels = True
    win._accept_result(result_payload(output))
    assert win.black.value() == 0 and win.white.value() == 255
    win.encoding.setCurrentText('tiff16')
    path = tmp_path/'normalised.tif'
    export_result(win.last_result, path, win._export_config())
    assert tifffile.imread(path).max() == pytest.approx(65535*.5, abs=1)


@pytest.mark.parametrize('mode', ['none', 'saturn'])
@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_cuda_normalisation_retains_same_sums(tmp_path, mode, colour):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_geometry_resume import capture, config as geometry_config
    path = capture(tmp_path, colour)
    cfg = replace(geometry_config(mode), device='gpu')
    with SERSource(path) as src:
        raw = stack_source(src, cfg)
        normal = stack_source(src, replace(cfg, normalise_brightness=True))
    assert raw.backend == normal.backend == 'cuda'
    gain = .7*65535/raw.image[raw.validity].max()
    np.testing.assert_allclose(normal.image, raw.image*gain, rtol=1e-14)
    np.testing.assert_array_equal(normal.coverage, raw.coverage)


def test_cli_enables_seventy_percent_output_normalisation(tmp_path, capsys):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path = write_ser(tmp_path/'input.ser', source()._frames)
    dest = tmp_path/'stack'
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--no-frame-preselection', '--no-local-alignment', '--normalise-brightness',
                 '--out', str(dest)]) == 0
    result = load_snapshot(dest/'stack.npz')
    assert result.image[result.validity].max() == pytest.approx(.7*255)
    assert result.provenance['brightness_normalisation']['percent'] == 70
    capsys.readouterr()
