"""Saturn CUDA must preserve region boundaries, raw colour support and resume."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from planetrecon.backends.torch_saturn import TorchSaturnWarp, TorchSaturnAccumulator
from planetrecon.detector import cfa_labels
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import SaturnSceneModel, MoonTrack, render_saturn
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=cuda)])
@pytest.mark.parametrize('latitude,transmission,field,surface,moon,sun', [
    (.18,.35,True,True,False,None), (-.4,0.,True,True,True,(.8,.1)),
    (.5,1.,False,True,True,(-.7,-.2)), (.4,.35,True,False,False,None),
    (.02,0.,True,True,False,None), (.4,0.,False,False,True,(.8,0.))])
def test_classification_projection_and_pull(device, latitude, transmission, field, surface, moon, sun):
    shape = (73,81)
    g = GlobeParams(17.,.12,.4,latitude,.2,.09,1.2)
    rings = RingParams(23.,34.,transmission,*(sun or (None,None)))
    model = SaturnSceneModel(g,rings,moon=MoonTrack(10,15,3,.4,-.2) if moon else None,
                              apply_field=field,apply_surface=surface,field_angle0_rad=.12)
    warp = TorchSaturnWarp(shape,model,device)
    src,ref = FramePose(7.,.13,40.2,35.6),FramePose(2.,-.1,41.3,36.2)
    x,y = detector_xy_grids(*shape)
    for pose in (src,ref):
        actual,expected = warp.classify(pose),model.classify_detector(x,y,pose)
        np.testing.assert_array_equal(actual['labels'].cpu(),expected['labels'])
        np.testing.assert_array_equal(actual['regions'].cpu(),model.reconstruction_regions(expected))
    actual,expected = warp.map(src,ref),model.src_to_ref(x,y,src,ref)
    np.testing.assert_array_equal(actual[2].cpu(),expected[2])
    for a,b in zip(actual[:2],expected[:2]):
        np.testing.assert_allclose(a.cpu(),b,atol=2e-11,rtol=0)
    image = np.random.default_rng(95).normal(size=(*shape,3))
    np.testing.assert_allclose(warp.render(image,src,ref),render_observed(image,model,src,ref),atol=2e-11,rtol=0)


def capture(colour='RGGB', latitude=.4, transmission=.35, moon=False):
    globe = GlobeParams(20.,.1,.3,latitude,0.,.07,1.5)
    rings = RingParams(26.,42.,transmission,.8,.2)
    track = MoonTrack(15,17,3,.3,-.1) if moon else None
    frames = []
    for t,(dx,dy) in enumerate([(0.,0.),(1.3,-2.4),(-2.,1.),(.3,.4)]):
        if moon:
            dx = dy = 0.
        image = render_saturn(112,112,FramePose(t,.025*(t-1.5)+.1,56+dx,56+dy),
            globe,rings,lambda lon,lat:1+.2*np.cos(5*lon)*np.cos(3*lat),moon=track,field_angle0_rad=.1)
        if colour in ('RGB','BGR'):
            image = image[...,None]*([.8,1.,.6] if colour == 'RGB' else [.6,1.,.8])
        elif colour != 'mono':
            labels = cfa_labels(112,112,colour)
            image *= np.where(labels=='R',.8,np.where(labels=='B',.6,1.))
        frames.append(image*1000)
    source = ArraySource(np.array(frames),color_mode=colour,timestamps=np.arange(4.))
    cfg = ReconstructionConfig(device='cpu',threads=2,frame_preselection=False,geometry_mode='saturn',
        field_center_x=56.,field_center_y=56.,equatorial_radius_px=20.,flattening=.1,pole_pa_rad=.3,
        sub_obs_lat_rad=latitude,surface_rate_rad_s=.07,field_rate_rad_s=.025,field_angle0_rad=.1,
        reference_epoch_s=1.5,ring_inner_radius_px=26.,ring_outer_radius_px=42.,ring_transmission=transmission,
        sun_lon_rad=.8,sun_lat_rad=.2,exposure_s=0.,batch_frames=2,
        **(dict(moon_x=15.,moon_y=17.,moon_radius_px=3.,moon_vx_px_s=.3,moon_vy_px_s=-.1) if moon else {}))
    return source,cfg


def equal_result(actual, expected):
    assert actual.n_used == expected.n_used and actual.n_rejected == expected.n_rejected
    np.testing.assert_array_equal(actual.validity,expected.validity)
    np.testing.assert_allclose(actual.image,expected.image,rtol=2e-9,atol=2e-7)
    np.testing.assert_allclose(actual.coverage,expected.coverage,rtol=2e-9,atol=2e-7)
    assert actual.layer_coverage.keys() == expected.layer_coverage.keys()
    for name in actual.layer_coverage:
        np.testing.assert_allclose(actual.layer_coverage[name],expected.layer_coverage[name],rtol=2e-9,atol=2e-7)


@cuda
@pytest.mark.parametrize('colour', ['mono','RGB','BGR','RGGB','GRBG','GBRG','BGGR'])
@pytest.mark.parametrize('latitude,transmission,moon', [(.18,.35,False),(-.4,0.,False),(.4,1.,True)])
def test_cuda_stack_matches_cpu(colour, latitude, transmission, moon):
    source,cfg = capture(colour,latitude,transmission,moon)
    cpu = stack_source(source,cfg)
    gpu = stack_source(source,replace(cfg,device='gpu'))
    assert gpu.backend == 'cuda' and gpu.n_used == 4
    assert not gpu.provenance['device_report']['fallback']
    equal_result(gpu,cpu)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=cuda)])
def test_identity_keeps_cfa_and_layer_support(device):
    shape=(65,71)
    model=SaturnSceneModel(GlobeParams(15.,sub_obs_lat_rad=.4),RingParams(20.,29.,0.))
    pose=FramePose(0.,0.,35.5,32.5)
    zeros=np.zeros((*shape,3));layer=np.zeros(shape)
    acc=TorchSaturnAccumulator(shape,model,'RGGB',zeros,zeros,zeros,zeros,layer,layer,pose,device)
    assert acc.add(np.ones(shape),pose,pose,1.)
    _,weight,_,_,globe,ring=acc.download()
    x,y=detector_xy_grids(*shape)
    info=model.classify_detector(x,y,pose)
    regions=model.reconstruction_regions(info)
    labels=cfa_labels(*shape,'RGGB')
    for i,c in enumerate('RGB'):
        np.testing.assert_array_equal(weight[...,i],(labels==c)&(regions>=0))
    np.testing.assert_array_equal(globe,(info['labels']==2)&(regions>=0))
    np.testing.assert_array_equal(ring,((info['labels']==1)|(info['labels']==3))&(regions>=0))


@cuda
def test_cancel_resume_restores_all_layers_and_detaches_previews(tmp_path, monkeypatch):
    from planetrecon.io.ser import SERSource, write_ser, NAME_TO_COLOR
    arrays,cfg=capture()
    path=write_ser(tmp_path/'saturn.ser',np.stack([arrays.read_raw(i) for i in range(4)]).astype('u2'),
                   color_id=NAME_TO_COLOR['RGGB'],timestamps=np.arange(4)*10000000)
    with SERSource(path) as source:
        cfg=replace(cfg,device='auto')
        whole=stack_source(source,cfg)
        state=tmp_path/'saturn.npz'
        saved=[]
        cancel=False
        def event(result,info):
            nonlocal cancel
            saved.append((result,result.image.copy(),{k:v.copy() for k,v in result.layer_coverage.items()}))
            cancel=info['n_processed']>=2
        partial=stack_source(source,cfg,state_checkpoint=state,on_event=event,should_cancel=lambda:cancel)
        assert partial.incomplete and partial.n_used==2
        continued=stack_source(source,cfg,resume_from=state)
        equal_result(continued,whole)
        for result,image,layers in saved:
            np.testing.assert_array_equal(result.image,image)
            for name,value in layers.items():
                np.testing.assert_array_equal(result.layer_coverage[name],value)
        with np.load(state) as z:
            assert json.loads(str(z['metadata']))['execution_history']==['cuda']
        from planetrecon.backends.base import DeviceReport
        monkeypatch.setattr('planetrecon.backends.base.probe_torch_cuda',lambda:
                            DeviceReport('auto','cpu',['cpu'],True,'cuda_not_available'))
        cpu_continued=stack_source(source,cfg,resume_from=state)
        equal_result(cpu_continued,whole)
        assert cpu_continued.provenance['execution_history']==['cuda','cpu']


def test_unavailable_cuda_retains_saturn_cpu(monkeypatch):
    from planetrecon.backends.base import DeviceReport
    monkeypatch.setattr('planetrecon.backends.base.probe_torch_cuda',lambda:
                        DeviceReport('gpu','cpu',['cpu'],True,'cuda_not_available'))
    source,cfg=capture()
    result=stack_source(source,replace(cfg,device='gpu'))
    assert result.backend=='cpu' and result.n_used==4
    assert 'cuda_not_available' in result.warnings
