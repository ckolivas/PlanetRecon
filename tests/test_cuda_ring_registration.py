"""Device ring tracking preserves CPU support, ambiguity and interpolation guards."""
import numpy as np
import pytest
from scipy.ndimage import shift

torch = pytest.importorskip('torch')
from planetrecon.backends.torch_registration import TorchRingRegistrationOps
from planetrecon.backends.torch_surface import TorchRingTracker
from planetrecon.geometry.model import FieldOnlyModel, render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.pipeline.ring_align import RingRegistration

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.fixture(params=['cpu', pytest.param('cuda:0', marks=cuda)])
def ops(request):
    return TorchRingRegistrationOps((112,128), request.param)


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
    return RingRegistration(image,64,56,15,48) if ops is None else TorchRingTracker(image,64,56,15,48,ops.device)


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


def test_batched_tracking_matches_single_frames(ops):
    image = reference()
    tracker = matcher(image, ops)
    ref = FramePose(0, 0, 64, 56)
    poses = [FramePose(k, .006 * k, 64 + .2 * k, 56 - .1 * k) for k in range(5)]
    frames = [shift(render_observed(image, FieldOnlyModel(), p, ref), (.3 * k - .5, .2 - .1 * k), order=3)
              for k, p in enumerate(poses)]
    frames[3] = np.ones_like(image)   # unconstrained: refused within the batch
    stacked = torch.as_tensor(np.stack(frames), device=ops.device)
    for batched, single in ((tracker.displacements_at_poses(stacked, poses, ref),
                             [tracker.displacement_at_pose(f, p, ref) for f, p in zip(frames, poses)]),
                            (tracker.displacements(stacked), [tracker.displacement(f) for f in frames])):
        assert [b is None for b in batched] == [s is None for s in single]
        assert batched[3] is None
        for b, s in zip(batched, single):
            if s is not None:
                np.testing.assert_allclose(b, s, atol=1e-12, rtol=0)
