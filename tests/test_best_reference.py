"""Choose the strongest retained preprocessing observation before accumulation."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess import FrameSelection
from planetrecon.pipeline.preprocess_cache import preprocess_source, load_cache
from planetrecon.reconstruction import ReconstructionConfig


def test_best_reference_ignores_rejected_scores_and_breaks_ties_by_index():
    scores = np.zeros((5, 4))
    scores[:, 0] = [1, 10, 99, 10, np.nan]
    selection = FrameSelection(np.array([True, True, False, True, False]), scores, {})
    assert selection.best_reference_index == 1
    selection.accepted[:] = False
    assert selection.best_reference_index is None


def capture(tmp_path):
    y, x = np.indices((64, 80))
    disc = gaussian_filter((((x-40)/21)**2+((y-32)/18)**2 < 1).astype(float), 1.)
    scene = (disc*(100+10*np.cos(y/2))).astype('u2')
    frames = np.stack([scene]*16)
    frames[11] *= 2  # Higher measured gradient score, identical apparent shape.
    return write_ser(tmp_path/'capture.ser', frames.astype('u2'))


@pytest.mark.parametrize('mode', ['none', 'field', 'surface', 'combined', 'saturn'])
def test_cached_best_reference_used_from_first_batch_and_after_resume(tmp_path, mode):
    path = capture(tmp_path)
    cfg = ReconstructionConfig(device='cpu', threads=2, batch_frames=2, geometry_mode=mode,
        cadence_s=.1, field_rate_rad_s=0., surface_rate_rad_s=0.,
        field_center_x=40.5, field_center_y=32.5, equatorial_radius_px=21.,
        sub_obs_lat_rad=.4, ring_inner_radius_px=25. if mode == 'saturn' else None,
        ring_outer_radius_px=30. if mode == 'saturn' else None)
    with SERSource(path) as source:
        selected = preprocess_source(source, cfg)
        assert selected.best_reference_index == 11
        assert selected.summary['geometry_estimate']['reference_index'] == 11
        _, report = load_cache(source, cfg)
        assert report['best_reference_index'] == 11
        whole = stack_source(source, cfg)
        assert whole.provenance['reference_index'] == 11
        assert cfg.reference_index == 0  # Automatic selection never mutates settings.
        explicit = stack_source(source, replace(cfg, reference_index=3))
        assert explicit.provenance['reference_index'] == 3
        bypass = stack_source(source, replace(cfg, frame_preselection=False))
        assert bypass.provenance['reference_index'] == 0
        stop = False
        def event(result, info):
            nonlocal stop
            if info['n_processed'] >= 4:
                assert result.provenance['reference_index'] == 11
                stop = True
        checkpoint = tmp_path/'resume.npz'
        partial = stack_source(source, cfg, on_event=event, should_cancel=lambda: stop,
                               state_checkpoint=checkpoint)
        assert partial.incomplete and partial.n_used == 4
        # The selected reference occurs after the interruption point.
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        assert resumed.provenance['reference_index'] == 11
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
