"""CUDA surface operators must preserve CPU radiometry, support and tracking."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from planetrecon.backends.torch_globe import TorchGlobeWarp, TorchGlobeAccumulator
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel, render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.pipeline.baseline import stack_source
from planetrecon.io.source import ArraySource
from test_globe_registration import sphere, config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=cuda)])
@pytest.mark.parametrize('rate', [0., -.07, .09])
@pytest.mark.parametrize('field', [False, True])
def test_projection_and_pull_match_cpu(device, rate, field):
    shape = (57,61)
    model = OblateGlobeModel(GlobeParams(23., .17, .4, -.3, .2, rate), apply_field=field)
    warp = TorchGlobeWarp(shape,model,device)
    src, ref = FramePose(7.,.13,28.2,27.3), FramePose(1.,-.2,31.1,26.9)
    x,y = detector_xy_grids(*shape)
    expected = model.src_to_ref(x,y,src,ref)
    actual = tuple(a.cpu().numpy() for a in warp.map(src,ref))
    np.testing.assert_array_equal(actual[2],expected[2])
    for a,b in zip(actual[:2],expected[:2]):
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=0)
    image = np.random.default_rng(143).normal(size=(*shape,3))
    np.testing.assert_allclose(warp.render(image,src,ref),render_observed(image,model,src,ref),atol=2e-11,rtol=0)


@cuda
@pytest.mark.parametrize('colour', ['mono','RGB','RGGB','GRBG','GBRG','BGGR'])
@pytest.mark.parametrize('field_rate', [0.,.04])
def test_cuda_stack_matches_cpu(colour, field_rate):
    from planetrecon.detector import cfa_labels
    frames = np.array([sphere(t,dx=dx,dy=dy,field_rate=field_rate)
                       for t,(dx,dy) in enumerate([(0,0),(1.3,-2.4),(-2.,1.),(.3,.4)])])
    if colour == 'RGB':
        frames = frames[...,None]*[.8,1.,.6]
    elif colour != 'mono':
        labels = cfa_labels(128,128,colour)
        frames *= np.where(labels=='R',.8,np.where(labels=='B',.6,1.))
    source = ArraySource(frames,color_mode=colour,bit_depth=32,timestamps=np.arange(4.))
    cfg = config(field_rate=field_rate)
    cpu = stack_source(source,cfg)
    gpu = stack_source(source,replace(cfg,device='gpu'))
    assert gpu.backend == 'cuda' and gpu.precision == 'float64'
    assert not gpu.provenance['device_report']['fallback']
    assert gpu.n_used == cpu.n_used == 4 and gpu.n_rejected == cpu.n_rejected == 0
    np.testing.assert_array_equal(gpu.validity,cpu.validity)
    np.testing.assert_allclose(gpu.coverage,cpu.coverage,atol=2e-8,rtol=2e-9)
    np.testing.assert_allclose(gpu.image,cpu.image,atol=2e-7,rtol=2e-9)


@cuda
def test_integer_identity_does_not_invent_bayer_coverage():
    from planetrecon.detector import cfa_labels
    shape=(31,35)
    model=OblateGlobeModel(GlobeParams(10.,surface_rate_rad_s=0.))
    zeros=np.zeros((*shape,3))
    accumulator=TorchGlobeAccumulator(shape,model,'RGGB',zeros,zeros,zeros,zeros)
    pose=FramePose(0.,0.,17.5,15.5)
    assert accumulator.add(np.ones(shape),pose,pose,1.)
    _,weight,_,_=accumulator.download()
    labels=cfa_labels(*shape,'RGGB')
    for i,c in enumerate('RGB'):
        np.testing.assert_array_equal(weight[...,i],labels==c)


@cuda
def test_cuda_cancel_resume_and_preview_ownership(tmp_path):
    from test_geometry_resume import capture, config as resume_config
    from planetrecon.io.ser import SERSource
    cfg=replace(resume_config('combined'),device='gpu',field_center_x=64.,field_center_y=64.,
                equatorial_radius_px=42.,sub_obs_lat_rad=0.,flattening=0.)
    path=capture(tmp_path,'RGGB','combined')
    state=tmp_path/'state.npz'
    with SERSource(path) as source:
        whole=stack_source(source,cfg)
    saved=[]
    cancel=False
    def event(result,info):
        nonlocal cancel
        if result.stage not in ('preprocessing','cache_ready'):
            saved.append((result,result.image.copy(),result.coverage.copy()))
            if info['n_processed'] >= 4: cancel=True
    with SERSource(path) as source:
        partial=stack_source(source,cfg,state_checkpoint=state,on_event=event,should_cancel=lambda:cancel)
    assert partial.incomplete and partial.n_used==3 and partial.n_rejected==1
    for result,image,coverage in saved:
        np.testing.assert_array_equal(result.image,image)
        np.testing.assert_array_equal(result.coverage,coverage)
    with SERSource(path) as source:
        continued=stack_source(source,cfg,resume_from=state)
    np.testing.assert_array_equal(continued.validity,whole.validity)
    np.testing.assert_allclose(continued.image,whole.image,rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose(continued.coverage,whole.coverage,rtol=1e-12,atol=1e-9)
    with np.load(state) as data:
        assert json.loads(str(data['metadata']))['execution_history']==['cuda']


def test_unavailable_cuda_falls_back_to_surface_cpu(monkeypatch):
    from planetrecon.backends.base import DeviceReport
    monkeypatch.setattr('planetrecon.backends.base.probe_torch_cuda',lambda:
                        DeviceReport('gpu','cpu',['cpu'],True,'cuda_not_available'))
    source=ArraySource(np.array([sphere(0),sphere(1)]),bit_depth=32,timestamps=np.arange(2.))
    result=stack_source(source,replace(config(),device='gpu'))
    assert result.backend=='cpu' and result.n_used==2
    assert 'cuda_not_available' in result.warnings


@cuda
@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB'])
@pytest.mark.parametrize('chunked', [False, True])
def test_fused_rank_bin_scatter_matches_per_bin_scatters(colour, chunked, monkeypatch):
    """One pass adds every frame to its rank bins exactly as separate scatters per bin do."""
    shape, n, n_bins = (37, 43), 5, 6
    model = OblateGlobeModel(GlobeParams(15., .1, .3, -.2, .1, .05), apply_field=True, ld_power=.4)
    rng = np.random.default_rng(171)
    channels = (*shape, 3) if colour != 'mono' else shape
    frames = rng.uniform(10, 200, (n, *shape, 3) if colour == 'RGB' else (n, *shape))
    demo = rng.uniform(10, 200, (n, *shape, 3)) if colour == 'RGGB' else None
    poses = [FramePose(float(k), .02*k, 21.3+.4*k, 18.1-.3*k) for k in range(n)]
    ref = FramePose(1.5, 0., 21., 18.)
    score = torch.as_tensor(rng.uniform(.5, 2., (n, *shape)), device="cuda")
    # Frames fall in different numbers of bins (none for frame 3).
    bins = [[(int(k), rng.uniform(0, 1, shape)) for k in rng.choice(n_bins, size, replace=False)]
            for size in (3, 1, 6, 0, 2)]
    results = []
    for fused in (True, False):
        zeros = np.zeros(channels)
        accumulator = TorchGlobeAccumulator(shape, model, colour, zeros, zeros,
                                            zeros if demo is not None else None, zeros if demo is not None else None)
        accumulator.enable_bins(np.zeros((n_bins, *channels)), np.zeros((n_bins, *channels)))
        if fused:
            assert accumulator.fused is not None
            if chunked:
                monkeypatch.setattr(accumulator, 'BIN_TABLE', 2*6*shape[0]*shape[1]*8)  # Two frames per call.
        else:
            accumulator.fused = None
        covered = accumulator.add(frames, poses, ref, score, demo, bins=bins)
        results.append((covered, *accumulator.download(), *accumulator.download_bins()))
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][1:], results[1][1:]):
        if a is not None:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12*np.abs(b).max())
    assert np.abs(results[0][-2]).sum(axis=tuple(range(1, results[0][-2].ndim))).min() > 0
