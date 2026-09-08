import numpy as np
import pytest
from tools.profile_scene_retained_cache import relative_error


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_profile_product_cannot_be_hidden_in_maximum(bad):
    errors=[relative_error(np.ones(2),np.ones(2)),relative_error(np.array([1.,bad]),np.ones(2))]
    assert max(errors)==np.inf


def test_finite_relative_profile_error():
    assert relative_error(np.array([3.,4.]),np.array([0.,4.]))==pytest.approx(.75)
