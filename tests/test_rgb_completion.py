import numpy as np
import pytest
import tifffile

from planetrecon.detector import cfa_labels
from planetrecon.export import export_result
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.colour import complete_bayer_rgb
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot


@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('geometry', ['none', 'field'])
def test_bayer_stack_publishes_and_exports_complete_rgb(pattern, geometry, tmp_path):
    labels = cfa_labels(9, 11, pattern)
    raw = np.zeros((9, 11))
    expected = np.array([110., 230., 70.])
    for c, name in enumerate('RGB'):
        raw[labels == name] = expected[c]
    source = ArraySource(raw[None], color_mode=pattern, bit_depth=32,
                         timestamps=np.array([0.]))
    cfg = ReconstructionConfig(frame_preselection=False, device='cpu', threads=2, geometry_mode=geometry,
                               field_rate_rad_s=0., field_center_x=5.5, field_center_y=4.5,
                               freeze_mid_exposure=False, reject_saturated=False)
    events = []
    result = stack_source(source, cfg, on_event=lambda r, _: events.append(r))
    for r in [result, *events]:
        assert r.channel_order == 'RGB'
        assert r.validity.all()
        np.testing.assert_allclose(r.image, np.broadcast_to(expected, r.image.shape))
        for name in 'RGB':
            np.testing.assert_array_equal(r.layer_coverage[f'cfa_direct_{name}'] > 0, labels == name)
        assert r.provenance['rgb_completion']['filled_channel_samples'] == 9*11*2
    save_snapshot(tmp_path/'result.npz', result)
    restored = load_snapshot(tmp_path/'result.npz')
    export_result(restored, tmp_path/'result.tif')
    pixels = tifffile.imread(tmp_path/'result.tif')
    assert np.isfinite(pixels).all()
    np.testing.assert_allclose(pixels, np.broadcast_to(expected, pixels.shape))


def test_completion_preserves_direct_samples_holes_and_region_boundaries():
    rng = np.random.default_rng(7)
    labels = cfa_labels(13, 15, 'RGGB')
    valid = np.stack([labels == c for c in 'RGB'], axis=2)
    valid[4:9, 4:9] = False
    image = np.where(valid, rng.uniform(1, 100, valid.shape), 0.)
    weight = valid * rng.uniform(1, 10, valid.shape)
    before_image, before_weight = image.copy(), weight.copy()
    regions = np.zeros((13, 15), dtype=int)
    regions[:, 8:] = 1
    r = ReconstructionResult(image, weight, valid, 'adu', 'RGB', 'cpu', 'float64',
                             'final', False, provenance={'color_mode': 'RGGB'})
    complete_bayer_rgb(r, regions)
    np.testing.assert_array_equal(weight, before_weight)
    np.testing.assert_array_equal(r.image[valid], before_image[valid])
    assert not r.validity[4:9, 4:9].any()
    for y, x, c in np.argwhere(r.validity & ~valid):
        sites = np.argwhere(valid[..., c] & (regions == regions[y, x]))
        sites = sites[np.max(np.abs(sites-[y, x]), axis=1) <= 1]
        distance = np.sum((sites-[y, x])**2, axis=1)
        nearest = sites[distance == distance.min()]
        assert r.image[y, x, c] in before_image[nearest[:, 0], nearest[:, 1], c]
    # A one-pixel region has no red/blue observations: never borrow across it.
    regions = np.arange(13*15).reshape(13, 15)
    r = ReconstructionResult(before_image.copy(), before_weight, valid, 'adu', 'RGB',
                             'cpu', 'float64', 'final', False, provenance={'color_mode': 'RGGB'})
    complete_bayer_rgb(r, regions)
    np.testing.assert_array_equal(r.validity, valid)


def test_completed_preview_does_not_fill_geometry_holes():
    from planetrecon.gui.preview import display_result_preview
    image = np.ones((7, 9, 3))
    valid = np.ones_like(image, dtype=bool)
    valid[2, 2] = False
    r = ReconstructionResult(image, valid.astype(float), valid, 'adu', 'RGB', 'cpu',
                             'float64', 'final', False,
                             provenance={'color_mode': 'RGGB', 'rgb_completion': {}})
    preview = display_result_preview(r, max_side=5)
    assert not preview.validity[1, 1].any()
