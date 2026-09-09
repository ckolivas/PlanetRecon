import json
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from planetrecon.pipeline import baseline
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.io.ser import SERSource,write_ser
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.detector import cfa_labels

@pytest.mark.parametrize('pattern',['RGGB','GRBG','GBRG','BGGR'])
def test_registration_luminance_uses_all_colours_without_a_cfa_pattern(pattern):
    labels=cfa_labels(81,95,pattern)
    raw=sum(value*(labels==name) for name,value in zip('RGB',[120.,150.,80.]))
    np.testing.assert_array_equal(baseline._colour_registration_plane(raw,pattern),np.full(raw.shape,125.))
    np.testing.assert_array_equal(baseline._alignment_plane(raw,pattern),np.full(raw.shape,150.))

def capture(path,shape=(96,112)):
    y,x=np.indices(shape);cx,cy=shape[1]/2,shape[0]/2;radius=min(shape)/3
    frames=[]
    for index in range(12):
        dx=.7*np.sin(index)
        image=gaussian_filter((((x-cx-dx)/radius)**2+((y-cy)/(radius*.9))**2<1)*(3000+300*np.cos((x-dx)/3)+400*np.sin(y/4)),.6)
        image[::2,::2]*=.8;image[1::2,1::2]*=.6
        frames.append(image)
    return write_ser(path,np.asarray(frames,dtype='u2'),color_id=8)

def config(**kwargs):
    return ReconstructionConfig(device='cpu',threads=2,batch_frames=2,**kwargs)

def test_colour_template_resume_is_exact_and_rejects_green_template_state(tmp_path):
    path=capture(tmp_path/'colour.ser');cfg=config(local_alignment=True)
    with SERSource(path) as source:
        selected=preprocess_source(source,cfg)
        cache=path.with_name(path.name+'.planetrecon-preprocess.npz');before=cache.read_bytes()
        whole=baseline.stack_source(source,cfg)
        assert whole.n_used==selected.accepted.sum()
        assert whole.provenance['local_alignment']['version']==2
        stop=False
        def event(result,info):
            nonlocal stop
            stop=info['n_processed']>=4
        checkpoint=tmp_path/'state.npz'
        partial=baseline.stack_source(source,cfg,on_event=event,should_cancel=lambda:stop,state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed=baseline.stack_source(source,cfg,resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image,whole.image)
        np.testing.assert_array_equal(resumed.coverage,whole.coverage)
        assert cache.read_bytes()==before
        with np.load(checkpoint) as data:
            payload={name:data[name].copy() for name in data.files}
        metadata=json.loads(str(payload['metadata']))
        assert metadata['identity']['local_registration_version']==2
        metadata['identity']['local_registration_version']=1
        payload['metadata']=json.dumps(metadata)
        np.savez_compressed(tmp_path/'old-state.npz',**payload)
        with pytest.raises(ValueError,match='identity'):
            baseline.stack_source(source,cfg,resume_from=tmp_path/'old-state.npz')

@pytest.mark.parametrize('local,shape',[(False,(96,112)),(True,(64,64))])
def test_global_and_small_capture_fallback_keep_green_registration(tmp_path,monkeypatch,local,shape):
    path=capture(tmp_path/'fallback.ser',shape);cfg=config(local_alignment=local)
    with SERSource(path) as source:
        selected=preprocess_source(source,cfg);assert selected.accepted.sum()>=4
        expected=baseline.stack_source(source,config())
        def forbidden(*args):
            pytest.fail('colour proxy must not alter ordinary global registration')
        monkeypatch.setattr(baseline,'_colour_registration_plane',forbidden)
        actual=baseline.stack_source(source,cfg)
        np.testing.assert_array_equal(actual.image,expected.image)
        np.testing.assert_array_equal(actual.coverage,expected.coverage)
