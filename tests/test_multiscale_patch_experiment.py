import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from tools.multiscale_patch_experiment import patch_totals, blend_totals, tiled_scores
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.backends.torch_circular import scores, peaks


def test_mixed_patch_sizes_reproduce_same_scene_without_smoothing():
    truth=np.random.default_rng(181).normal(size=(43,49))*12+100
    total,weight=np.zeros_like(truth),np.zeros_like(truth)
    for size,centres in [(9,[(15,16),(21,22)]),(17,[(17,20),(25,29)]),(25,[(21,25)])]:
        h=size//2
        patches=np.array([truth[y-h:y+h+1,x-h:x+h+1] for y,x in centres])
        y,x=np.indices((size,size))-h
        foot=np.maximum(0.,1.-np.hypot(x,y)/h)
        a,b=patch_totals(patches,np.ones_like(patches),centres,foot,truth.shape)
        total+=a; weight+=b
    combined=blend_totals(total,weight,truth)
    np.testing.assert_allclose(combined,truth,rtol=0,atol=1e-12)
    assert weight.max()>2 and weight.min()==0


def test_mixed_scale_support_preserves_absolute_brightness_at_edges():
    fallback=np.full((41,43),137.5)
    total,weight=np.zeros_like(fallback),np.zeros_like(fallback)
    for w,centre in [(11,(17,18)),(25,(23,24))]:
        support=np.random.default_rng(w).uniform(size=(1,w,w))
        support[:,0,:]=0
        a,b=patch_totals(np.full((1,w,w),137.5),support,[centre],np.ones((w,w)),fallback.shape)
        total+=a; weight+=b
    np.testing.assert_allclose(blend_totals(total,weight,fallback),137.5,atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for tiled NCC parity')
@pytest.mark.parametrize('size',[65,93,131,185])
def test_tiled_large_patch_scores_match_existing_ncc(size):
    rng=np.random.default_rng(331)
    reference=gaussian_filter(rng.normal(size=(256,304)),1.5)*10+100
    layer=LocalRegistration(reference,window=size,step=size//2,circular=True)
    centres=np.array([(y,x) for y in layer.ys for x in layer.xs])[:2]
    image=torch.as_tensor(layer.ref,device='cuda')
    y=torch.as_tensor(centres[:,0],device='cuda'); x=torch.as_tensor(centres[:,1],device='cuda')
    template=torch.as_tensor(layer.templates[:len(centres)],device='cuda')
    strength=torch.as_tensor(layer.strength[:len(centres)],device='cuda')
    weight=torch.as_tensor(layer.weight,device='cuda')
    h=size//2+3
    py,px=torch.meshgrid(torch.arange(-h,h+1,device='cuda'),torch.arange(-h,h+1,device='cuda'),indexing='ij')
    expected=scores(image[y[:,None,None]+py,x[:,None,None]+px],template,strength,weight)
    actual=tiled_scores(image,y,x,template,strength,weight)
    np.testing.assert_allclose(actual.cpu().numpy(),expected.cpu().numpy(),rtol=0,atol=2e-12)
    offsets,valid,_=peaks(actual); wanted,good,_=peaks(expected)
    np.testing.assert_array_equal(valid.cpu(),good.cpu())
    np.testing.assert_allclose(offsets.cpu(),wanted.cpu(),atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for sampling parity')
@pytest.mark.parametrize('coordinate_dtype',[torch.float32,torch.float64])
def test_fused_sampling_matches_original_with_broadcasts_and_exterior(coordinate_dtype):
    from tools.multiscale_patch_experiment import sample, reference_sample
    rng=np.random.default_rng(682)
    image=torch.as_tensor(rng.normal(size=(2,17,19))*100,device='cuda')
    y=torch.tensor([[[-.7],[3.2],[16.4]],[[.3],[8.7],[18.]]],device='cuda',dtype=coordinate_dtype)
    x=torch.tensor([[[-1.2,0.,5.7,18.3]]],device='cuda',dtype=coordinate_dtype)
    actual=sample(image,y,x); expected=reference_sample(image,y,x)
    np.testing.assert_allclose(actual.cpu(),expected.cpu(),rtol=0,atol=1e-12)


def test_quality_remains_eligible_when_only_motion_search_crosses_edge():
    from tools.multiscale_patch_experiment import supported_windows
    centres=np.array([[68,133]])  # 131px AP, 3px original border margin.
    assert supported_windows(centres,(404,696),(0.,-1.5),65).all()
    assert not supported_windows(centres,(404,696),(0.,-1.5),68).any()
    assert not supported_windows(centres,(404,696),(0.,-4.),65).any()
