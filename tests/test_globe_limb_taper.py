"""Spin may only move light that is clearly on the globe, never seeing-blurred light across the limb."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel, deposition_model
from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig

R, C, RATE = 50., 64., .002


def frame(t):
    """An equator-on limb-darkened sphere spinning about a vertical axis, blurred by seeing."""
    y, x = np.indices((128, 128), dtype=float)
    sx, sy = x+.5-C, C-y-.5
    inside = sx*sx + sy*sy < R*R
    z = np.sqrt(np.maximum(R*R - sx*sx - sy*sy, 0.))
    lon, lat = np.arctan2(sx, z) + RATE*t, np.arcsin(np.clip(sy/R, -1, 1))
    texture = 100 + 25*np.cos(9*lon)*np.cos(3*lat) + 15*np.sin(10*lat) + 15*np.sin(17*lon - 5*lat)
    return gaussian_filter(np.where(inside, texture*(z/R)**.4, 0.), 2.) + 2


def devices():
    out = ['cpu']
    try:
        import torch
        if torch.cuda.is_available():
            out.append('gpu')
    except ImportError:
        pass
    return out


def limb_rind(taper, device, monkeypatch):
    monkeypatch.setattr(OblateGlobeModel, 'deposit_limb_taper_mu', taper)
    times = np.linspace(-150, 150, 31)
    cfg = ReconstructionConfig(device=device, threads=2, frame_preselection=False, geometry_mode='surface',
                               field_center_x=C, field_center_y=C, equatorial_radius_px=R, surface_rate_rad_s=RATE,
                               limb_darkening_power=0., reference_epoch_s=150.)
    result = stack_source(ArraySource(np.stack([frame(t) for t in times]), timestamps=times), cfg)
    truth = frame(0.)
    y, x = np.indices(truth.shape) + .5
    r = np.hypot(x - C, y - C)
    assert np.sqrt(np.mean((result.image - truth)[r < .8*R]**2)) < .7  # The surface itself is derotated.
    band = (r > R-3) & (r < R-1) & (abs(y - C) < .3*R)  # Just inside both limbs along the equator.
    return float(np.mean(result.image[band] - truth[band])), result


@pytest.mark.parametrize('device', devices())
def test_blurred_limb_light_is_not_moved_into_a_bright_rind(device, monkeypatch):
    split, _ = limb_rind(None, device, monkeypatch)
    tapered, result = limb_rind(.4, device, monkeypatch)
    assert split > 10  # A hard boundary drags the blurred limb inwards: rind, then a dark arc when sharpened.
    assert abs(tapered) < 4
    assert result.provenance['geometry']['limb_taper']['applies_to'] == 'sample deposition'


def test_tracking_predictions_keep_the_exact_map():
    model = OblateGlobeModel(GlobeParams(42, surface_rate_rad_s=.1))
    deposit = deposition_model(model)
    assert model.limb_taper_mu is None and deposit.limb_taper_mu == pytest.approx(.4)
    y, x = np.indices((128, 128), dtype=float) + .5
    src, ref = FramePose(3., 0., 64, 64), FramePose(0., 0., 64, 64)
    ex, ey, ev = model.src_to_ref(x, y, src, ref)
    tx, ty, tv = deposit.src_to_ref(x, y, src, ref)
    r = np.hypot(x - 64, y - 64)
    deep = r < 30  # mu > 0.7 at both epochs: the taper leaves the surface map untouched.
    np.testing.assert_allclose(tx[deep], ex[deep], atol=1e-12)
    assert not ev[(r < 41) & ~deep].all()  # The exact map drops light rotating out of view...
    assert tv[r < 41].all()  # ...which the deposit keeps in field alignment.
