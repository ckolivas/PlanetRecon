"""The geometry silhouette must not become a new photometric edge."""
import numpy as np
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import SaturnSceneModel
from planetrecon.geometry.pose import FramePose


def test_spin_displacement_is_continuous_across_the_limb():
    model=SaturnSceneModel(GlobeParams(90.,surface_rate_rad_s=.00016,sub_obs_lat_rad=.18),RingParams(130.,210.))
    src,ref=FramePose(180.,0.,256.,160.),FramePose(0.,0.,256.,160.)
    # Oversample the equatorial limb: a boundary crossing must not cause a
    # whole-pixel jump, including a newly visible surface on either side.
    x=np.linspace(160.,352.,19201);y=np.full_like(x,160.)
    mx,my,valid=model.src_to_ref(x,y,src,ref)
    assert valid.all()
    assert np.max(np.abs(np.diff(mx-x))) < .015
    assert np.max(np.abs(np.diff(my-y))) < .015
    np.testing.assert_allclose(mx[[0,-1]],x[[0,-1]],atol=1e-12)
    np.testing.assert_allclose(my[[0,-1]],y[[0,-1]],atol=1e-12)


def test_ring_edges_and_shadows_do_not_gate_observed_samples():
    from planetrecon.geometry.coords import detector_xy_grids
    model=SaturnSceneModel(GlobeParams(20.,surface_rate_rad_s=.07,sub_obs_lat_rad=.18),RingParams(28.,45.))
    x,y=detector_xy_grids(112,112)
    mx,my,valid=model.src_to_ref(x,y,FramePose(2.,.08,56.3,55.6),FramePose(0.,0.,56.,56.))
    assert valid.all() and np.isfinite(mx).all() and np.isfinite(my).all()


def test_limb_has_no_jump_in_displacement_derivative():
    model=SaturnSceneModel(GlobeParams(90.,surface_rate_rad_s=.00016,sub_obs_lat_rad=.18),RingParams(130.,210.))
    src,ref=FramePose(180.,0.,256.,160.),FramePose(0.,0.,256.,160.)
    for edge in (166.,346.):
        x=edge+np.linspace(-.002,.002,41);y=np.full_like(x,160.)
        mx,my,_=model.src_to_ref(x,y,src,ref)
        assert np.max(np.abs(np.diff(mx-x)/np.diff(x))) < .005
        assert np.max(np.abs(np.diff(my-y)/np.diff(x))) < .005
