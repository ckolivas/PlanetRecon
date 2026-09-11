"""CUDA ring tracking preserves CPU support, ambiguity and interpolation guards."""
import numpy as np
import pytest
from scipy.ndimage import shift
from scipy.signal import correlate

torch = pytest.importorskip('torch')
from planetrecon.backends.torch_registration import TorchRingRegistrationOps
from planetrecon.geometry.model import FieldOnlyModel, render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.pipeline.ring_align import RingRegistration

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.fixture(params=['cpu', pytest.param('cuda:0', marks=cuda)])
def ops(request):
    return TorchRingRegistrationOps((112,128), request.param)


def test_linear_correlations_and_complete_observed_support(ops):
    rng = np.random.default_rng(132)
    proxy = rng.normal(size=(112,128)) + 10
    mask = np.zeros(proxy.shape)
    mask[21:77, 28:101] = 1
    mask[34:46, 45:60] = 0
    template = rng.normal(size=proxy.shape)*mask
    count, energy = mask.sum(), np.sum(template**2)
    dot, total, squares, support = [correlate(a,b,mode='full',method='fft') for a,b in (
        (proxy,template),(proxy,mask),(proxy**2,mask),(np.ones(proxy.shape),mask))]
    denominator = np.sqrt(energy*np.maximum(squares-total**2/count,0))
    valid = (support >= count-1e-6) & (denominator > 1e-12)
    actual = ops.scores(proxy,template,mask,count,energy)
    np.testing.assert_array_equal(np.isfinite(actual),valid)
    np.testing.assert_allclose(actual[valid],dot[valid]/denominator[valid],atol=1e-11,rtol=1e-10)


def test_field_pulls_match_at_edges_and_fractional_centres(ops):
    image = np.random.default_rng(23).normal(size=(112,128))
    src,ref = FramePose(2,.12,63.3,55.7),FramePose(0,-.08,64.,56.)
    for a,b in ((src,ref),(ref,src),(src,src)):
        np.testing.assert_allclose(ops.render(image,a,b),
            render_observed(image,FieldOnlyModel(),a,b),atol=1e-12,rtol=0)


def reference():
    y,x = np.indices((112,128))
    radius = np.hypot((x-64)/1.3,(y-56)/.6)
    return np.exp(-((radius-32)/5)**2)


def matcher(image, ops=None):
    return RingRegistration(image,64,56,15,48,
        **({} if ops is None else dict(score_provider=ops.scores,renderer=ops.render)))


@pytest.mark.parametrize('kind', ['fractional','brightness','field','flat','repeated'])
def test_alignment_and_refusal_match_cpu(ops,kind):
    image = reference()
    if kind == 'flat':
        image = np.ones_like(image)
    elif kind == 'repeated':
        image = 1+np.cos(np.indices(image.shape)[1]*2*np.pi/8)
    frame = shift(image,(1.3,-2.4),order=3,mode='constant')
    if kind == 'brightness':
        frame = .7*frame+12
    cpu,gpu = matcher(image),matcher(image,ops)
    if kind == 'field':
        ref,pose = FramePose(0,0,64,56),FramePose(1,.018,64,56)
        frame = shift(render_observed(image,FieldOnlyModel(),pose,ref),(.3,-.4),order=3)
        expected,actual = cpu.displacement_at_pose(frame,pose,ref),gpu.displacement_at_pose(frame,pose,ref)
    else:
        expected,actual = cpu.displacement(frame),gpu.displacement(frame)
    if kind in ('flat','repeated'):
        assert expected is None and actual is None
    else:
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=0)
