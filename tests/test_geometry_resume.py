from dataclasses import replace
import json
import numpy as np
import pytest

from planetrecon.io.ser import write_ser, SERSource, COLOR_RGGB, COLOR_RGB
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def capture(tmp_path, color, mode=None):
    if mode in ('surface', 'combined'):
        from test_globe_registration import sphere
        from planetrecon.detector import cfa_labels
        frames = np.array([sphere(i*.2, rate=.08, dx=i, field_rate=.04 if mode == 'combined' else 0.)
                           for i in range(7)])*100
        if color == 'RGB':
            frames = frames[..., None]*[.8, 1., .6]
        elif color == 'RGGB':
            labels = cfa_labels(128, 128, color)
            frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
        frames = frames.astype('u2')
        frames[2] = 65535
        return write_ser(tmp_path/'in.ser', frames,
                         color_id={'mono':0, 'RGB':COLOR_RGB, 'RGGB':COLOR_RGGB}[color])
    y,x=np.indices((40,48))
    frame=(100+600*np.exp(-((x-23.5)**2+(y-19.5)**2)/80)+20*np.cos(x/2)).astype('u2')
    if color=='RGB':frame=np.stack([frame,frame//2,frame//3],axis=-1)
    frames=np.stack([np.roll(frame,i,axis=1) for i in range(7)])
    frames[2]=65535
    return write_ser(tmp_path/'in.ser',frames,color_id={'mono':0,'RGB':COLOR_RGB,'RGGB':COLOR_RGGB}[color])


def config(mode):
    return ReconstructionConfig(device='cpu',threads=2,batch_frames=2,geometry_mode=mode,
        cadence_s=.2,reference_epoch_s=.15,field_center_x=24.,field_center_y=20.,
        equatorial_radius_px=9.,sub_obs_lat_rad=.4,flattening=.1,
        field_rate_rad_s=.04,surface_rate_rad_s=.08,
        ring_inner_radius_px=12. if mode=='saturn' else None,
        ring_outer_radius_px=19. if mode=='saturn' else None)


@pytest.mark.parametrize('mode',['field','surface','combined','saturn'])
@pytest.mark.parametrize('color',['mono','RGB','RGGB'])
def test_geometry_continuation_is_exact(tmp_path,mode,color):
    path=capture(tmp_path,color,mode);cfg=config(mode);state=tmp_path/'state.npz'
    if mode in ('surface', 'combined'):
        cfg=replace(cfg,field_center_x=64.,field_center_y=64.,equatorial_radius_px=42.,
                    sub_obs_lat_rad=0.,flattening=0.)
    with SERSource(path) as src:whole=stack_source(src,cfg)
    cancel=False
    def event(result,info):
        nonlocal cancel
        if result.stage != 'preprocessing' and info['n_processed']>=4:cancel=True
    with SERSource(path) as src:
        partial=stack_source(src,cfg,state_checkpoint=state,on_event=event,should_cancel=lambda:cancel)
    assert partial.incomplete and partial.n_used==3 and partial.n_rejected==1
    with SERSource(path) as src:continued=stack_source(src,cfg,resume_from=state,state_checkpoint=state)
    assert continued.provenance['resumed_from_frame']==4
    assert continued.n_used==whole.n_used and continued.n_rejected==whole.n_rejected
    for key in ('image','coverage','validity'):
        np.testing.assert_array_equal(getattr(whole,key),getattr(continued,key))
    for key in whole.layer_coverage:
        np.testing.assert_array_equal(whole.layer_coverage[key],continued.layer_coverage[key])
    with SERSource(path) as src:again=stack_source(src,cfg,resume_from=state)
    np.testing.assert_array_equal(again.image,whole.image)


def test_geometry_identity_and_layer_validation(tmp_path):
    path=capture(tmp_path,'RGGB');cfg=config('saturn');state=tmp_path/'state.npz'
    with SERSource(path) as src:stack_source(src,cfg,state_checkpoint=state)
    with SERSource(path) as src:
        with pytest.raises(ValueError,match='mismatch'):
            stack_source(src,replace(cfg,field_rate_rad_s=.05),resume_from=state)
    with np.load(state,allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    arrays['ring_weight'][0,0]=-1
    np.savez(state,**arrays)
    with SERSource(path) as src:
        with pytest.raises(ValueError,match='negative checkpoint weights'):
            stack_source(src,cfg,resume_from=state)


def test_cancel_before_geometry_preserves_checkpoint(tmp_path):
    path=capture(tmp_path,'mono');cfg=config('combined');state=tmp_path/'state.npz'
    state.write_bytes(b'previous')
    with SERSource(path) as src:
        result=stack_source(src,cfg,state_checkpoint=state,should_cancel=lambda:True)
    assert result.incomplete and result.n_used==0
    assert state.read_bytes()==b'previous'
