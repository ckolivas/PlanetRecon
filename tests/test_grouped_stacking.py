"""Grouped CUDA stacking pieces: fused kernels, deferred progress results and grouped ranking."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.io.ser import SERSource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from tests.test_ap_align import capture, points_config


def cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_ap import fused_kernels
    if fused_kernels() is None:
        pytest.skip('Triton unavailable')
    return torch


@pytest.mark.parametrize('color', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_fused_pull_matches_the_torch_group_backprojection(color):
    torch = cuda()
    from planetrecon.backends import triton_stack
    from planetrecon.backends.torch_accel import TorchBackend
    from planetrecon.detector import BAYER_PATTERNS
    rng = np.random.default_rng(11)
    shape = (3, 29, 35, 3) if color == 'RGB' else (3, 29, 35)
    images = torch.as_tensor(rng.uniform(0, 300, shape), device='cuda')
    fields = torch.as_tensor(rng.normal(0, 4, (3, 2, 29, 35)), device='cuda')  # Some samples leave the frame.
    labels = ([["RGB".index(c) for c in row] for row in BAYER_PATTERNS[color]] if color in BAYER_PATTERNS else None)
    expected = TorchBackend().backproject_group(images, fields, color)
    np.testing.assert_allclose(triton_stack.pull(images, fields, labels).cpu().numpy(), expected.cpu().numpy(),
                               rtol=1e-14, atol=1e-12)


def test_fused_accumulation_matches_per_term_sums():
    torch = cuda()
    from planetrecon.backends import triton_stack
    from planetrecon.backends.torch_ap import DeviceSums
    rng = np.random.default_rng(12)
    h, w, K, G, D = 23, 31, 7, 3, 6
    blend = (torch.as_tensor(rng.random((h, K)), device='cuda'), torch.as_tensor(rng.random((K, w)), device='cuda'),
             torch.as_tensor(rng.random(h * w), device='cuda'), torch.as_tensor(rng.random(h * w), device='cuda'))
    contributions = torch.as_tensor(rng.random((G, h * w, D)), device='cuda')
    pairs = [(0, 4, np.array([0, 3]), 0., 2.), (1, 4, np.array([1]), 1., .5), (2, 1, np.array([], int), 1., 3.),
             (0, 2, np.arange(K), 0., 1.5), (2, 4, np.array([6, 2, 5]), 1., 1.)]
    fused = DeviceSums((h, w), 3, 4, blend)
    assert fused.kernels is triton_stack
    fused.add([10, 11, 12], contributions, pairs)
    plain = DeviceSums((h, w), 3, 4, blend)
    plain.kernels = None
    plain.add([10, 11, 12], contributions, pairs)
    np.testing.assert_allclose(fused.sums.cpu().numpy(), plain.sums.cpu().numpy(), rtol=1e-13, atol=1e-13)
    assert fused.pending == [10, 11, 12] and sorted(fused.bin_pending) == [10, 12] and fused.touched == {1, 2}


def test_progress_results_form_only_when_read_or_kept(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = points_config(normalise_brightness=True)
        preprocess_source(source, cfg)
        unread, kept = [], []

        def event(result, info):
            if result.stage == 'baseline':
                unread.append(result.formed)
                kept.append((result, info['n_used']))
        final = stack_source(source, cfg, on_event=event)
    # Progress alone formed nothing; results the receiver keeps are formed before the sums move on.
    assert unread and not any(unread)
    assert all(result.formed for result, _ in kept)
    totals = [result.coverage.sum() for result, _ in kept]
    assert totals == sorted(totals) and totals[0] < totals[-1]
    last, used = kept[-1]
    assert used == final.n_used and 'rgb_completion' not in last.provenance  # Mono: nothing to complete.
    assert last.provenance['brightness_normalisation']['applied']


def test_grouped_ranking_matches_frame_by_frame_ranking():
    from planetrecon.pipeline.ap_align import AlignmentPoints, LocalQuality, select_locally
    from tests.test_ap_align import planet
    rng = np.random.default_rng(5)
    ref = planet()
    frames = [ref + rng.normal(0, 1, ref.shape) for _ in range(11)]
    matcher = AlignmentPoints(ref, size=48)
    shift = lambda r, p: (float(np.mean(p) % .3), -.1)
    quality = LocalQuality(matcher)
    mask = np.ones(11, bool)

    def measure(indices):
        found = [shift(ref, frames[i]) for i in indices]
        return found, np.stack([quality.measure(frames[i], s) for i, s in zip(indices, found)])
    one = select_locally(matcher, np.arange(11), lambda i: frames[i], shift, ref, 5, mask.copy())
    grouped = select_locally(matcher, np.arange(11), lambda i: frames[i], shift, ref, 5, mask.copy(),
                             measure_group=measure, group=4)
    np.testing.assert_array_equal(one.chosen, grouped.chosen)
    assert one.shifts == grouped.shifts


def test_cuda_groups_match_the_cpu_with_fourier_weighting_and_moons(tmp_path):
    cuda()
    with SERSource(capture(tmp_path, color_id=8)) as source:
        cfg = replace(points_config(), fourier_weighting=True, batch_frames=7)
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        gpu = stack_source(source, replace(cfg, device='gpu'))
    execution = gpu.provenance['local_alignment']['cuda_execution']
    assert execution['frame_group'] >= 2 and 'Triton' in execution['accumulation']
    assert gpu.n_used == cpu.n_used
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-7, atol=1e-7 * cpu.image.max())
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('color_id', [0, 8])
def test_saturated_frames_are_rejected_on_the_device_as_on_the_cpu(tmp_path, color_id, monkeypatch):
    """Screening normally drops saturated frames first; here it does not, so stacking's own test decides."""
    cuda()
    import planetrecon.pipeline.baseline as baseline
    from planetrecon.io.ser import write_ser
    with SERSource(capture(tmp_path, color_id=color_id)) as source:
        frames = np.stack([source.read_raw(i) for i in range(source.n_frames())])
    frames[[3, 9]] = np.where(np.arange(frames.shape[2])[None, :] < frames.shape[2] // 8, 65535, frames[[3, 9]])
    path = write_ser(tmp_path / 'saturated.ser', frames, color_id=color_id)
    with SERSource(path) as source:
        cfg = replace(points_config(), stack_percent=100)
        with monkeypatch.context() as patch:
            patch.setattr(baseline, '_saturated', lambda frame, bit_depth: False)
            preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        gpu = stack_source(source, replace(cfg, device='gpu'))
    assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used and gpu.n_rejected == cpu.n_rejected >= 2
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-7, atol=1e-7 * cpu.image.max())
