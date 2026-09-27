import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from tools.coherent_registration import SplineField, CoherentRegistration
from tools.validate_local_warp_centring import make_frame


def fitter_scene():
    y,x = np.mgrid[12:120:12, 12:152:12]
    points = np.column_stack((x.ravel(), y.ravel()))
    return SplineField((132,164), points, spacing=16), points


def test_fit_retains_true_constant_offset_and_affine_geometry():
    fitter, points = fitter_scene()
    values = np.column_stack((.8+.002*points[:,0], -.5+.003*points[:,1]))
    field, stats = fitter.fit(values, np.ones(len(points)))
    y,x = np.indices(fitter.shape)
    expected = np.stack((.8+.002*x, -.5+.003*y))
    assert not stats['fallback']
    np.testing.assert_allclose(field[:,20:-20,20:-20], expected[:,20:-20,20:-20], atol=1e-4)


def test_robust_fit_rejects_isolated_bad_measurements():
    fitter, points = fitter_scene()
    values = np.tile([.8,-.5], (len(points),1))
    values[::11] += [3.,-3.]
    field, stats = fitter.fit(values, np.ones(len(points)))
    assert stats['downweighted_points'] >= len(points)//11
    assert np.sqrt(np.mean((field[:,20:-20,20:-20]-np.array([.8,-.5])[:,None,None])**2)) < .15


def test_insufficient_measurements_fall_back_to_global_residual():
    fitter, points = fitter_scene()
    field, stats = fitter.fit(np.ones((len(points),2)), np.zeros(len(points)))
    assert stats['fallback']
    np.testing.assert_array_equal(field, 0.)


def test_flat_reference_retains_global_translation():
    reference = np.ones((40,48))*20
    engine = CoherentRegistration(CircularMultiscaleRegistration(reference), device='cpu')
    result = engine.displacement(LocalRegistration.proxy(reference), (.3,-.2), lambda: None)
    np.testing.assert_array_equal(result.numpy()[0], .3)
    np.testing.assert_array_equal(result.numpy()[1], -.2)
    assert engine.stats[-1]['fallback']


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('patch_average', [False,True])
def test_static_image_has_no_spline_warp(device,patch_average):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        reference = 100+30*gaussian_filter(np.random.default_rng(51).normal(size=(96,112)),1.)
        engine = CoherentRegistration(CircularMultiscaleRegistration(reference), device=device, spacing=16,
                                      patch_average=patch_average)
        result = engine.displacement(LocalRegistration.proxy(reference), (0.,0.), lambda: None)
        np.testing.assert_allclose(result.cpu().numpy(), 0., atol=1e-12)
    finally:
        torch.set_num_threads(previous)


def test_patch_average_basis_retains_affine_measurements():
    reference = 100+30*gaussian_filter(np.random.default_rng(27).normal(size=(80,96)),1.)
    engine = CoherentRegistration(CircularMultiscaleRegistration(reference), device='cpu',spacing=16,patch_average=True)
    fitter = engine.fitter
    cy,cx = np.meshgrid((np.arange(fitter.ny)-1)*16,(np.arange(fitter.nx)-1)*16,indexing='ij')
    coefficients = (.8+.002*cx-.003*cy).ravel()
    observed = fitter.design@coefficients
    expected = []
    for layer,kernel in zip(reversed(engine.layers),engine.kernels):
        x = layer['x'][:,None,None]+layer['px']
        y = layer['y'][:,None,None]+layer['py']
        expected.extend(((.8+.002*x-.003*y)*kernel).sum((-2,-1)).numpy())
    np.testing.assert_allclose(observed,expected,atol=1e-12)


def test_moving_patch_average_cpu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        reference = 100+30*gaussian_filter(np.random.default_rng(94).normal(size=(96,112)),1.)
        frame,_,_ = make_frame(reference,.8,-.3,0.)
        fields = []
        for device in ('cpu','cuda'):
            engine = CoherentRegistration(CircularMultiscaleRegistration(reference),device=device,
                                          spacing=16,stiffness=.01,patch_average=True)
            fields.append(engine.displacement(LocalRegistration.proxy(frame),(.1,-.1),lambda:None).cpu().numpy())
            assert engine.stats[-1]['field_guard_accepted']
        np.testing.assert_allclose(fields[0],fields[1],atol=1e-8)
        assert np.std(fields[0])>.05
    finally:
        torch.set_num_threads(previous)
