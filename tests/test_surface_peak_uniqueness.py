"""Broad surface correlation shoulders are not distinct drift solutions."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.ndimage import shift

from planetrecon.pipeline.masked_align import MaskedRegistration


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('offset', [(0., 0.), (2.3, -1.7), (-4., 3.)])
def test_unique_broad_texture_recovers_translation_at_multiple_samplings(scale, offset):
    y, x = np.indices((160*scale, 160*scale), dtype=float)
    x, y = (x-80*scale)/scale, (y-80*scale)/scale
    reference = 100*np.exp(-((x/40)**2+(y/27)**2)) + 8*np.exp(-(((x-25)/12)**2+((y+12)/9)**2))
    mask = x*x+y*y < 55**2
    frame = shift(reference, (offset[1], offset[0]), order=3, mode='constant')
    registration = MaskedRegistration(reference, mask)
    assert registration.displacement(frame) is None  # Old fixed-width uniqueness rejects shoulders.
    actual = registration.displacement(frame, distinct_peaks=True)
    assert actual is not None
    np.testing.assert_allclose(actual, offset, atol=.02)


@pytest.mark.parametrize('kind', ['flat', 'bands', 'repeated'])
def test_unconstrained_or_repeated_texture_is_still_refused(kind):
    y, x = np.indices((192, 192), dtype=float)
    image = np.ones_like(x) if kind == 'flat' else np.cos(y/5)
    if kind == 'repeated':
        image += np.cos(x/5)
    mask = (x-96)**2 + (y-96)**2 < 45**2
    assert MaskedRegistration(image, mask).displacement(image, distinct_peaks=True) is None


@pytest.mark.parametrize('radius', [60, 120])
@pytest.mark.parametrize('rate,limb_weight', [(0., True), (.002, False)])
def test_broad_planetary_bands_and_feature_recover_surface_drift(radius, rate, limb_weight):
    from planetrecon.geometry.globe import GlobeParams, render_globe_texture
    from planetrecon.geometry.pose import FramePose
    from planetrecon.geometry.model import OblateGlobeModel
    from planetrecon.pipeline.globe_align import surface_displacement
    n = radius*3
    globe = GlobeParams(radius, surface_rate_rad_s=rate, sub_obs_lat_rad=.04)
    model = OblateGlobeModel(globe)
    def texture(lon, lat):
        return 100+15*np.sin(4*lat)+12*np.exp(-((lon-.3)**2+(lat+.1)**2)/.08)
    # The rigid image warp transports radiance, not a changing illumination
    # law: test broad limb shading without spin and body-fixed texture with spin.
    reference = render_globe_texture(n,n,n/2,n/2,0.,globe,0.,texture,limb_weight=limb_weight)
    frame = render_globe_texture(n,n,n/2+2.2,n/2-1.4,0.,globe,40.,texture,limb_weight=limb_weight)
    actual = surface_displacement(reference,frame,model,FramePose(40.,0.,n/2,n/2),FramePose(0.,0.,n/2,n/2))
    assert actual is not None
    np.testing.assert_allclose(actual,(2.2,-1.4),atol=.12)


def test_surface_checkpoint_binds_peak_uniqueness_policy(tmp_path):
    from test_globe_registration import sphere, config
    from planetrecon.io.ser import write_ser, SERSource
    from planetrecon.pipeline.baseline import stack_source
    path = write_ser(tmp_path/'in.ser', (np.array([sphere(i) for i in range(3)])*100).astype('u2'))
    checkpoint = tmp_path/'state.npz'
    cfg = replace(config(), cadence_s=1.)
    with SERSource(path) as source:
        whole = stack_source(source,cfg,state_checkpoint=checkpoint)
        restored = stack_source(source,cfg,resume_from=checkpoint)
        np.testing.assert_array_equal(whole.image,restored.image)
        with np.load(checkpoint) as state:
            payload = {key:state[key].copy() for key in state.files}
        metadata = json.loads(str(payload['metadata']))
        assert metadata['identity']['geometry'].pop('surface_peak_uniqueness')
        payload['metadata'] = json.dumps(metadata)
        np.savez_compressed(tmp_path/'old.npz',**payload)
        with pytest.raises(ValueError,match='identity/configuration'):
            stack_source(source,cfg,resume_from=tmp_path/'old.npz')
