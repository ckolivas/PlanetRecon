import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline import baseline, local_align
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig


def capture(path, colour):
    yy, xx = np.indices((96, 112))
    disk = ((xx-56)/35)**2 + ((yy-48)/32)**2 < 1
    scene = disk*(3000+300*np.cos(xx/2)+400*np.sin(yy/3))
    frames = np.stack([gaussian_filter(scene, sigma) for sigma in np.linspace(.6, 1.4, 12)])
    if colour == 8:
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .6
    elif colour == 100:
        frames = frames[..., None]*[.8, 1., .6]
    return write_ser(path, frames.astype('u2'), color_id=colour)


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=2,
                                local_alignment=True, **kwargs)


@pytest.mark.parametrize('colour', [0, 8, 100])
def test_standard_quality_weights_match_independent_sums(tmp_path, monkeypatch, colour):
    path = capture(tmp_path/'weights.ser', colour)
    with SERSource(path) as source:
        cfg = config()
        selection = preprocess_source(source, cfg)
        projections = []
        original = local_align.cpu_backproject
        def record(raw, shift, color):
            projected = original(raw, shift, color)
            projections.append((projected[0].copy(), projected[1].copy()))
            return projected
        monkeypatch.setattr(local_align, 'cpu_backproject', record)
        stacked = baseline.stack_source(source, cfg)
        q = selection.measurements[selection.accepted, 0]
        assert len(projections) == stacked.n_used == len(q)
        numerator = np.zeros_like(stacked.image)
        denominator = np.zeros_like(stacked.coverage)
        for quality, (add, support) in zip(q, projections):
            numerator += quality*add
            denominator += quality*support
        expected = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        native = denominator > 0
        np.testing.assert_allclose(stacked.image[native], expected[native], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(stacked.coverage[native], denominator[native])
        assert stacked.provenance['scalar_frame_weight'] == 'linear quality'


def test_standard_checkpoint_resumes_and_rejects_legacy_squared_sums(tmp_path):
    path = capture(tmp_path/'resume.ser', 8)
    cfg = config()
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = baseline.stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            stop = info['n_processed'] >= 4
        checkpoint = tmp_path/'state.npz'
        partial = baseline.stack_source(source, cfg, on_event=event,
                                        should_cancel=lambda: stop, state_checkpoint=checkpoint)
        assert partial.incomplete
        with np.load(checkpoint) as data:
            payload = {key:data[key].copy() for key in data.files}
        metadata = json.loads(str(payload['metadata']))
        for settings in (metadata['identity']['config'], metadata['identity']['capture']['config']):
            # Standard checkpoints have always omitted the disabled option.
            assert 'squared_quality_weights' not in settings
        resumed = baseline.stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        for settings in (metadata['identity']['config'], metadata['identity']['capture']['config']):
            settings['squared_quality_weights'] = True
        payload['metadata'] = json.dumps(metadata)
        np.savez_compressed(tmp_path/'legacy-squared.npz', **payload)
        with pytest.raises(ValueError, match='identity|configuration'):
            baseline.stack_source(source, cfg, resume_from=tmp_path/'legacy-squared.npz')


def test_saved_standard_settings_load_without_removed_option():
    expected = config()
    saved = {**expected.to_dict(), 'squared_quality_weights': False}
    assert ReconstructionConfig.from_dict(saved) == expected
    assert saved['squared_quality_weights'] is False
    assert 'squared_quality_weights' not in expected.to_dict()


@pytest.mark.parametrize('value', [True, 1, None])
def test_saved_stronger_weight_settings_are_not_silently_reinterpreted(value):
    saved = {**config().to_dict(), 'squared_quality_weights': value}
    with pytest.raises(ValueError, match='Stronger quality weighting was removed'):
        ReconstructionConfig.from_dict(saved)


def test_cli_no_longer_offers_stronger_weighting(capsys):
    from planetrecon.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['stack', '--help'])
    assert exc.value.code == 0
    assert '--squared-quality-weights' not in capsys.readouterr().out
