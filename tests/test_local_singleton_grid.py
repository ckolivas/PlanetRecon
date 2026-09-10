"""Single-row/column local matchers must cover the capture centre symmetrically."""
import json
import numpy as np
import pytest

from planetrecon.pipeline.local_align import LocalRegistration, patch_centres, pull


def scene(y,x):
    return 300+30*np.cos(x/3)+35*np.sin(y/4)+20*np.sin((x+y)/7)


@pytest.mark.parametrize('shape', [(97,97),(97,160),(160,97)])
def test_singleton_axes_match_under_reflection(shape):
    y,x=np.indices(shape,dtype=float)
    ref=scene(y,x);frame=scene(y-.5,x-.75)
    flow=LocalRegistration(ref).displacement(frame,(0.,0.))
    for axis,length in enumerate(shape):
        if length != 97:continue
        flipped=LocalRegistration(np.flip(ref,axis)).displacement(np.flip(frame,axis),(0.,0.))
        for component in range(2):
            sign=-1 if component == 1-axis else 1
            np.testing.assert_allclose(np.flip(flipped[component],axis),sign*flow[component],atol=2e-12)


def test_central_detail_gets_local_correction_when_only_one_patch_fits():
    y,x=np.indices((97,97),dtype=float)
    ref=scene(y,x);frame=scene(y-.5,x-.75)
    matcher=LocalRegistration(ref)
    np.testing.assert_array_equal(matcher.ys,[48])
    np.testing.assert_array_equal(matcher.xs,[48])
    flow=matcher.displacement(frame,(0.,0.))
    np.testing.assert_allclose([flow[0][48,48],flow[1][48,48]],[.75,.5],atol=.03)
    roi=np.s_[43:54,43:54]
    assert np.linalg.norm((pull(frame,flow)-ref)[roi]) < .4*np.linalg.norm((frame-ref)[roi])


@pytest.mark.parametrize('length', [71,72,73,96,97,102])
def test_singleton_grid_keeps_complete_search_footprint(length):
    centre=patch_centres(length,65,32)
    assert len(centre)==1 and centre[0]==(length-1)//2
    assert centre[0]-35>=0 and centre[0]+35<length


@pytest.mark.parametrize('length', [103,160,424,656])
def test_multi_patch_axes_remain_unchanged(length):
    np.testing.assert_array_equal(patch_centres(length,65,32),np.arange(35,length-35,32))


def test_precentering_checkpoint_is_refused(tmp_path):
    from test_local_alignment import capture,config
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    with SERSource(capture(tmp_path)) as source:
        cfg=config();preprocess_source(source,cfg)
        checkpoint=tmp_path/'new.npz'
        whole=stack_source(source,cfg,state_checkpoint=checkpoint)
        assert whole.provenance['local_alignment']['patch_grid']=='singleton axes centred'
        with np.load(checkpoint) as data:payload={key:data[key].copy() for key in data.files}
        metadata=json.loads(str(payload['metadata']))
        assert metadata['identity'].pop('local_patch_grid')=='singleton axes centred'
        payload['metadata']=json.dumps(metadata)
        old=tmp_path/'old.npz';np.savez_compressed(old,**payload)
        with pytest.raises(ValueError,match='identity'):
            stack_source(source,cfg,resume_from=old)
        resumed=stack_source(source,cfg,resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image,whole.image)


@pytest.mark.hardware
@pytest.mark.parametrize('shape', [(97,97),(97,160),(160,97)])
def test_offset_cuda_tiles_match_cpu(shape):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    y,x=np.indices(shape,dtype=float)
    ref=scene(y,x);frame=scene(y-.5,x-.75)
    cpu=LocalRegistration(ref).displacement(frame,(.125,-.25))
    gpu=LocalRegistration(ref,use_cuda=True).displacement(frame,(.125,-.25))
    np.testing.assert_allclose(cpu,gpu,atol=1e-10,rtol=1e-10)
    assert max(np.max(np.abs(v)) for v in cpu)>.3
