"""Independent detector-scale detail must not alias into false rotation."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def fine_scene(angle, scale=1.):
    size = int(256*scale)
    y, x = np.indices((size, size), dtype=float)
    radius = np.hypot(x+.5-size/2, size/2-y-.5)
    theta = np.arctan2(size/2-y-.5, x+.5-size/2)-angle
    # Direct evaluation of a continuous scene. Fine outer features retain
    # roughly 6.7 detector pixels per cycle as the image size changes; broader
    # asymmetric features disambiguate the repeated fine structure.
    image = (radius < 110*scale)*(1
        + .2*np.exp(-((radius-85*scale)/(15*scale))**2)*np.cos(int(80*scale)*theta)
        + .04*np.cos(3*theta)+.03*np.sin(2*theta))
    return image, radius


@pytest.mark.parametrize('scale', [.5, 1., 2.])
@pytest.mark.parametrize('degrees', [-2., -1., -.5, .5, 1., 2.])
def test_fine_angular_detail_recovers_rotation_across_image_sizes(scale, degrees):
    reference, _ = fine_scene(0., scale)
    image, _ = fine_scene(np.deg2rad(degrees), scale)
    result = estimate_field_angle(reference, image, 128*scale, 128*scale, 110*scale)
    assert not result['degeneracy']
    assert np.rad2deg(result['angle_rad']) == pytest.approx(degrees, abs=.03)


@pytest.mark.parametrize('direction', [-1., 1.])
def test_fine_angular_detail_improves_unsharpened_stack(direction):
    rate = direction*np.deg2rad(1)/8
    frames = np.array([fine_scene(rate*i)[0] for i in range(9)])
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=128, field_center_y=128,
        equatorial_radius_px=110)
    corrected = stack_source(src, cfg)
    zero = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    oracle = stack_source(src, replace(cfg, field_rate_rad_s=rate))
    _, radius = fine_scene(0.)
    interior = (radius > 55) & (radius < 100)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        assert result.validity[interior].all()
        return np.sqrt(np.mean((result.image[interior]-frames[0][interior])**2))
    assert corrected.provenance['geometry']['field_rate_rad_s'] == pytest.approx(rate, rel=.02)
    assert error(corrected) < .5*error(zero)
    assert error(corrected) < 1.1*error(oracle)
