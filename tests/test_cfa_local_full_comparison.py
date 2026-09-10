import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from tools.cfa_local_full_comparison import fit_pair


def test_full_comparison_uses_common_mask_and_detects_detail_loss():
    y, x = np.indices((96, 112))
    disk = ((x-56)/40)**2+((y-48)/35)**2 < 1
    scene = disk*(1000+100*np.cos(x/3)+80*np.sin(y/4))
    target = scene[..., None]*[.8, 1., .6]
    ordinary = gaussian_filter(target, (2, 2, 0))
    metrics, previews, mask = fit_pair(ordinary, target, target, np.ones(target.shape, bool))
    assert metrics['passes_both_metrics']
    assert metrics['common_pixels'] == mask.sum() > 100
    assert metrics['interpolated']['relative_rmse'] < 1e-12
    assert set(previews) == {'ordinary', 'interpolated'}
    equal, _, repeated = fit_pair(target, target, target, np.ones(target.shape, bool))
    np.testing.assert_array_equal(mask, repeated)
    assert not equal['passes_both_metrics']


def test_full_comparison_refuses_insufficient_shared_support():
    reference = np.ones((96, 112, 3))
    with pytest.raises(ValueError, match='insufficient'):
        fit_pair(reference, reference, reference, np.zeros(reference.shape, bool))
