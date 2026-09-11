"""Motion predictions and seeing corrections compose before raw samples move."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter,map_coordinates

from planetrecon.pipeline.motion_local import inverse_local_coordinates,local_coordinates
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import SaturnSceneModel
from planetrecon.geometry.model import render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.warp import bilinear_push


def test_inverse_residual_composes_spatially_varying_motion():
    y,x=np.indices((128,160),dtype=float)
    ux,uy=2*np.sin(y/24)*np.sin(x/30),1.5*np.cos(x/22)*np.sin(y/30)
    qx,qy=inverse_local_coordinates((ux,uy));qx-=.5;qy-=.5
    np.testing.assert_allclose(qx+map_coordinates(ux,[qy,qx],order=1,mode='nearest'),x,atol=1e-7)
    np.testing.assert_allclose(qy+map_coordinates(uy,[qy,qx],order=1,mode='nearest'),y,atol=1e-7)
    assert np.max(abs((x-ux)+map_coordinates(ux,[y-uy,x-ux],order=1,mode='nearest')-x)) > .03


def test_unobserved_prediction_does_not_create_patch_motion():
    ref=gaussian_filter(np.random.default_rng(7).normal(size=(100,120)),1.)
    shift=LocalRegistration(ref,window=33,step=16,valid_mask=np.zeros(ref.shape,bool)).displacement(np.roll(ref,2,axis=1),(0.,0.))
    assert all(not np.any(axis) for axis in shift)


def test_predicted_rotation_is_retained_while_local_seeing_is_corrected():
    shape=(160,192);y,x=np.indices(shape,dtype=float)
    template=gaussian_filter(np.random.default_rng(204).normal(size=shape),1.5)*100+300
    model=SaturnSceneModel(GlobeParams(52.,surface_rate_rad_s=.05,sub_obs_lat_rad=.3),RingParams(65.,90.))
    anchor,pose=FramePose(0.,0.,96.,80.),FramePose(1.,.02,96.,80.)
    render=lambda image,src,ref:render_observed(image,model,src,ref)
    predicted=render(template,pose,anchor)
    identity=local_coordinates(template,np.ones(shape,bool),predicted,pose,anchor,render,33,False)
    np.testing.assert_allclose(identity[0],x+.5,atol=1e-10)
    np.testing.assert_allclose(identity[1],y+.5,atol=1e-10)
    flow=1.5*np.sin(y/25)*np.sin(x/45)
    observed=map_coordinates(predicted,[y,x-flow],order=3,mode='reflect')
    qx,qy=local_coordinates(template,np.ones(shape,bool),observed,pose,anchor,render,33,False)
    total,weight=bilinear_push(observed,qy-.5,qx-.5,shape)
    corrected=np.divide(total,weight,out=np.zeros_like(total),where=weight>0)
    roi=np.s_[48:-48,48:-48]
    assert np.linalg.norm((corrected-predicted)[roi]) < .65*np.linalg.norm((observed-predicted)[roi])


def selected(source,cfg):
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.reconstruction import ReconstructionConfig
    result=screen_source(source,ReconstructionConfig(frame_preselection=False,reject_saturated=False))
    result.accepted[:]=True
    result.measurements[:,0]=np.arange(source.n_frames(),0,-1)
    from planetrecon.pipeline.preprocess_cache import identity,selection_digest
    result.identity=identity(source,cfg,None)
    result.digest=selection_digest(result)
    return result


@pytest.mark.parametrize('mode',['saturn','surface','combined'])
@pytest.mark.parametrize('colour',['mono','RGGB'])
def test_pipeline_local_motion_changes_samples_and_resumes(tmp_path,mode,colour):
    from planetrecon.io.ser import SERSource,write_ser,NAME_TO_COLOR
    from planetrecon.pipeline.baseline import stack_source
    if mode=='saturn':
        from test_cuda_saturn import capture
        src,cfg=capture(colour)
        frames=np.stack([src.read_raw(i) for i in range(src.n_frames())])
    else:
        from test_globe_registration import sphere,config
        from planetrecon.detector import cfa_labels
        cfg=config(rate=.04,field_rate=.01 if mode=='combined' else 0.)
        frames=np.stack([sphere(i,rate=.04,field_rate=.01 if mode=='combined' else 0.) for i in range(4)])*10
        if colour=='RGGB':
            labels=cfa_labels(128,128,colour);frames*=np.where(labels=='R',.8,np.where(labels=='B',.6,1.))
    path=write_ser(tmp_path/'motion.ser',frames.astype('u2'),color_id=NAME_TO_COLOR[colour],timestamps=np.arange(4)*10000000)
    cfg=replace(cfg,frame_preselection=True,local_alignment=True,local_patch_size=33,stack_percent=100,exposure_s=0.,batch_frames=2)
    state=tmp_path/'state.npz'
    with SERSource(path) as src:
        selection=selected(src,cfg)
        whole=stack_source(src,cfg,preprocessing=selection)
        ordinary=stack_source(src,replace(cfg,local_alignment=False),preprocessing=selection)
        assert whole.n_used==ordinary.n_used==4
        assert not np.array_equal(whole.image,ordinary.image)
        stop=False
        def event(r,info):
            nonlocal stop
            stop=info['n_processed']>=2 if r.stage!='cache_ready' else False
        partial=stack_source(src,cfg,preprocessing=selection,on_event=event,should_cancel=lambda:stop,state_checkpoint=state)
        assert partial.incomplete
        resumed=stack_source(src,cfg,preprocessing=selection,resume_from=state)
        np.testing.assert_array_equal(resumed.image,whole.image)
        np.testing.assert_array_equal(resumed.coverage,whole.coverage)
        with pytest.raises(ValueError,match='identity/configuration mismatch'):
            stack_source(src,replace(cfg,local_patch_size=49),preprocessing=selection,resume_from=state)


@pytest.mark.parametrize('colour',['mono','RGB','RGGB'])
@pytest.mark.parametrize('moon',[False,True])
def test_cuda_motion_local_matches_cpu(colour,moon):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_cuda_saturn import capture,equal_result
    from planetrecon.pipeline.baseline import stack_source
    src,cfg=capture(colour,moon=moon)
    cfg=replace(cfg,frame_preselection=True,local_alignment=True,local_patch_size=33,stack_percent=100)
    selection=selected(src,cfg)
    cpu=stack_source(src,cfg,preprocessing=selection)
    gpu=stack_source(src,replace(cfg,device='gpu'),preprocessing=selection)
    equal_result(gpu,cpu)


def test_gui_keeps_patch_choice_for_saturn_and_surface():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    from planetrecon.reconstruction import ReconstructionConfig
    app=QApplication.instance() or QApplication([])
    controls=ConfigControls(ReconstructionConfig(local_alignment=True))
    try:
        geometry=controls.fields['geometry_mode']
        for mode in ('saturn','surface','combined','none'):
            geometry.setCurrentIndex(geometry.findData(mode))
            assert controls.fields['local_alignment'].isEnabled()
            assert controls.configuration().local_alignment
        controls.fields['local_alignment'].setChecked(False)
        assert not controls.configuration().local_alignment
    finally:
        controls.close()


def test_cuda_local_resume_can_continue_on_cpu(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_cuda_saturn import capture,equal_result
    from planetrecon.io.ser import SERSource,write_ser,NAME_TO_COLOR
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.backends.base import DeviceReport
    arrays,cfg=capture('RGGB')
    path=write_ser(tmp_path/'local.ser',np.stack([arrays.read_raw(i) for i in range(4)]).astype('u2'),
        color_id=NAME_TO_COLOR['RGGB'],timestamps=np.arange(4)*10000000)
    cfg=replace(cfg,device='auto',frame_preselection=True,local_alignment=True,local_patch_size=33,stack_percent=100)
    with SERSource(path) as source:
        selection=selected(source,cfg)
        whole=stack_source(source,cfg,preprocessing=selection)
        state=tmp_path/'state.npz';stop=False
        def event(result,info):
            nonlocal stop
            stop=result.stage!='cache_ready' and info.get('n_processed',0)>=2
        partial=stack_source(source,cfg,preprocessing=selection,state_checkpoint=state,on_event=event,should_cancel=lambda:stop)
        assert partial.incomplete and partial.n_used==2
        resumed=stack_source(source,cfg,preprocessing=selection,resume_from=state)
        equal_result(resumed,whole)
        monkeypatch.setattr('planetrecon.backends.base.probe_torch_cuda',lambda:
            DeviceReport('auto','cpu',['cpu'],True,'cuda_not_available'))
        cpu=stack_source(source,cfg,preprocessing=selection,resume_from=state)
        equal_result(cpu,whole)
        assert cpu.provenance['execution_history']==['cuda','cpu']
