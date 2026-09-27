import pytest

from tools.ap_noise_ladder import LEVELS, POLICIES, key
from tools.analyse_ap_noise_ladder import effects,crossings


def test_factorial_reports_conditional_effects_without_hiding_interaction():
    result=effects(dict(baseline=5.,cubic=3.,refined=4.,cubic_refined=6.))
    assert result==dict(cubic_with_original_peak=-2.,cubic_with_refined_peak=2.,
        refinement_with_bilinear=-1.,refinement_with_cubic=3.,interaction=4.)


def test_noise_brackets_preserve_multiple_crossings_and_unique_cell_ids():
    brackets=crossings([0,.25,.5,1,2],[-1.,1.,-2.,-3.,4.])
    assert [(x['lower_sigma'],x['upper_sigma']) for x in brackets]==[(0,.25),(.25,.5),(1,2)]
    assert crossings([0,1,2],[-1,0,1])==[]
    with pytest.raises(ValueError):
        crossings([0,1,1],[-1,0,1])
    assert len({key(p,n) for p in POLICIES for n in LEVELS})==20
