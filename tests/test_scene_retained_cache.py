import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_fft import SceneFFTBatch
from tools.scene_retained_cache import RetainedSceneFFTBatch, spectrum_storage


@pytest.mark.parametrize('rgb', [False, True])
def test_sequential_reuse_preserves_exact_arithmetic_and_memory_bound(rgb):
    rng = np.random.default_rng(314)
    shape = (8, 8, 3) if rgb else (8, 8)
    ops = []
    for _ in range(4):
        h = rng.uniform(size=(3, 4)); h /= h.sum()
        ops.append(SceneDetectorOperator(shape, (h,), 2, (1, 1), (3, 3),
                                          cfa_pattern='RGGB' if rgb else None))
    size = spectrum_storage(shape, (3, 4), 4)
    budget = 2*size['per_spectrum_bytes']
    retained = RetainedSceneFFTBatch(ops, cache_bytes=budget)
    lru = SceneFFTBatch(ops, cache_bytes=budget)
    x = rng.uniform(size=shape); weights = [rng.uniform(size=op.output_shape) for op in ops]
    for _ in range(3):
        np.testing.assert_array_equal(retained.normal(x, weights), lru.normal(x, weights))
    assert retained.hits == 4 and retained.misses == 8
    assert lru.hits == 0 and lru.misses == 12
    assert retained.resident_bytes == budget and retained.peak_cache_bytes <= budget
    assert list(retained.cache) == [0, 1]
    retained.clear_cache(); assert retained.resident_bytes == 0 and not retained.cache


def test_full_sequence_size_accounts_for_shared_rgb_psf_and_complex128():
    mono = spectrum_storage((1024, 912), (512, 512), 500)
    rgb = spectrum_storage((1024, 912, 3), (512, 512), 500)
    assert mono == rgb
    assert mono['fft_shape'] == [1536, 1440]
    assert mono['all_spectra_bytes'] == 8859648000


def test_tiny_cache_keeps_nothing():
    op = SceneDetectorOperator((4, 4), (np.ones((1, 1)),), 1, (0, 0), (4, 4))
    batch = RetainedSceneFFTBatch([op], cache_bytes=1)
    for _ in range(2): np.testing.assert_allclose(batch.forward(np.ones((4, 4)))[0], 1.)
    assert batch.resident_bytes == 0 and batch.hits == 0 and batch.misses == 2


@pytest.mark.hardware
@pytest.mark.parametrize('factor', [1, 2])
def test_retained_cuda_cells_match_independent_spatial_cpu(factor):
    import torch
    from tools.scene_study import CellBasisOperator
    from tools.scene_retained_cache import RetainedCellFFTBatch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(315)
    ops = []
    for _ in range(4):
        h = rng.uniform(size=(3, 4)); h /= h.sum()
        base = SceneDetectorOperator((8, 8, 3), (h,), 2, (1, 1), (3, 3), cfa_pattern='GBRG')
        ops.append(CellBasisOperator(base, factor))
    budget = 2*spectrum_storage((8, 8, 3), (3, 4), 4)['per_spectrum_bytes']
    batch = RetainedCellFFTBatch(ops, device='cuda', cache_bytes=budget)
    x = rng.uniform(size=ops[0].scene_shape)
    weights = [rng.uniform(size=op.output_shape) for op in ops]
    expected = sum(op.adjoint(w*op.forward(x)) for op,w in zip(ops,weights))
    for _ in range(3): np.testing.assert_allclose(batch.normal(x,weights), expected, rtol=1e-12, atol=1e-12)
    assert batch.cache_info()['hits'] == 4 and batch.cache_info()['peak_bytes'] == budget
