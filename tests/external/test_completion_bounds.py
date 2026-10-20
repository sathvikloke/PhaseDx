"""The eight pre-specified Batch26 completion-bound cases, preserved."""
import numpy as np
import pytest
from phasedx_external.scorer26 import completion_bounds

N = float("nan")
Y = [1, 1, 0, 0]
G = [.5, .4, .6, .3]


@pytest.mark.parametrize("y,a,b,expected", [
    (Y, [.9,.6,.5,.1], G, [0.5,0.5]),
    (Y, [.9,N,.5,.1], G, [-0.25,0.75]),
    (Y, [.9,.6,.5,N], G, [0.,1.]),
    (Y, [.9,N,.5,N], G, [-0.5,1.]),
    (Y, [.9,N,.5,.1], [.5,.4,.6,N], [-0.5,1.]),
    (Y, [.5,.5,.5,.1], G, [0.25,0.25]),
    ([1,0], [N,.4], [.7,N], [-1.,1.]),
    ([1,1], [.4,.5], [.1,.2], None),
])
def test_prespecified_completion_case(y, a, b, expected):
    a, b = np.array(a), np.array(b)
    result = completion_bounds(np.array(y), a, b, np.isfinite(a) & np.isfinite(b))
    assert result["N_pos"] + result["N_neg"] == len(y)
    if expected is None:
        assert result["status"] == "undefined_T0" and result["bounds"] is None
    else:
        assert result["bounds"]["difference"] == expected
