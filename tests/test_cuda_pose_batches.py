"""Batched device geometry equals the same poses processed one at a time."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from planetrecon.backends.torch_globe import TorchGlobeAccumulator, TorchGlobeWarp
from planetrecon.backends.torch_registration import TorchRingRegistrationOps
from planetrecon.backends.torch_saturn import TorchSaturnAccumulator, TorchSaturnWarp
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import MoonTrack, SaturnSceneModel

devices = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                 reason='CUDA unavailable'))]
SHAPE = (73, 81)
# Includes a pose at the reference time, whose mapping has no surface rotation.
POSES = [FramePose(7., .13, 40.2, 35.6), FramePose(2., -.1, 41.3, 36.2), FramePose(-3., .02, 39.7, 36.9)]
REF = FramePose(2., -.1, 41.3, 36.2)


def saturn(moon=False):
    g = GlobeParams(17., .12, .4, .18, .2, .09, 1.2)
    return SaturnSceneModel(g, RingParams(23., 34., .35), moon=MoonTrack(10, 15, 3, .4, -.2) if moon else None,
                            apply_field=True, apply_surface=True, field_angle0_rad=.12)


def globe():
    return OblateGlobeModel(GlobeParams(17., .12, .4, .18, .2, .09, 1.2), apply_field=True, apply_surface=True)


def same(batched, singles):
    # CUDA divides by a host scalar through its reciprocal: 1 ulp from a column divisor.
    for b, single in zip(batched, singles):
        if b.dtype == torch.bool or not b.is_floating_point():
            np.testing.assert_array_equal(b.cpu().numpy(), single.cpu().numpy())
        else:
            np.testing.assert_allclose(b.cpu().numpy(), single.cpu().numpy(), rtol=0, atol=1e-12)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('make', [globe, saturn, lambda: saturn(moon=True)])
def test_maps_and_renders(device, make):
    warp = (TorchSaturnWarp if isinstance(make(), SaturnSceneModel) else TorchGlobeWarp)(SHAPE, make(), device)
    for src, ref in ((POSES, REF), (REF, POSES)):
        batched = warp.map(src, ref)
        singles = [warp.map(s, r) for s, r in zip(*(p if isinstance(p, list) else [p] * 3 for p in (src, ref)))]
        for k in range(3):
            same(batched[k], [s[k] for s in singles])
    image = torch.as_tensor(np.random.default_rng(3).normal(size=(*SHAPE, 2)), device=device)
    same(warp.render_tensor(image, POSES, REF), [warp.render_tensor(image, p, REF) for p in POSES])
    frames = torch.as_tensor(np.random.default_rng(4).normal(size=(3, *SHAPE)), device=device)
    same(warp.render_tensor(frames, REF, POSES, per_pose=True),
         [warp.render_tensor(f, REF, p) for f, p in zip(frames, POSES)])
    if isinstance(warp, TorchSaturnWarp):
        batched = warp.classify(POSES)
        for key in batched:
            same(batched[key], [warp.classify(p)[key] for p in POSES])


@pytest.mark.parametrize('device', devices)
def test_ring_field_map(device):
    ops = TorchRingRegistrationOps(SHAPE, device)
    for src, ref in ((POSES, REF), (REF, POSES)):
        batched = ops.map(src, ref)
        pairs = [(p, REF) for p in POSES] if src is POSES else [(REF, p) for p in POSES]
        for k in range(3):
            same(batched[k], [ops.map(*pair)[k] for pair in pairs])


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('kind', ['globe', 'saturn'])
@pytest.mark.parametrize('local', [False, True])
def test_batched_accumulation(device, kind, local):
    rng = np.random.default_rng(8)
    h, w = SHAPE

    def sums():
        return [np.zeros((h, w, 3)), np.zeros((h, w, 3)), np.zeros((h, w, 3)), np.zeros((h, w, 3))]
    if kind == 'globe':
        make = lambda: TorchGlobeAccumulator(SHAPE, globe(), 'RGGB', *sums(), device)
    else:
        make = lambda: TorchSaturnAccumulator(SHAPE, saturn(), 'RGGB', *sums(), np.zeros(SHAPE), np.zeros(SHAPE),
                                              REF, device)
    one, many = make(), make()
    frames = torch.as_tensor(rng.uniform(1, 2, (3, h, w)), device=device)
    demos = torch.as_tensor(rng.uniform(1, 2, (3, h, w, 3)), device=device)
    scores = torch.as_tensor(rng.uniform(.5, 1, (3, h, w)) if local else rng.uniform(.5, 1, 3), device=device)
    sample = None
    if local:
        y, x = np.indices(SHAPE) + .5
        sample = tuple(torch.as_tensor(np.stack([c + .3 * rng.standard_normal(SHAPE) for _ in range(3)]),
                                       device=device) for c in (x, y))
    for k, pose in enumerate(POSES):
        one.add(frames[k], pose, REF, scores[k] if local else float(scores[k]), demos[k],
                None if sample is None else (sample[0][k], sample[1][k]))
    assert many.add(frames, POSES, REF, scores, demos, sample) == [True] * 3
    for a, b in zip(one.download(), many.download()):
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-13)
