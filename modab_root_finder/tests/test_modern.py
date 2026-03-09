from modab_root_finder import root_scalar
from modab_root_finder.modern import (
    _secant,
    _sign,
)
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, assume, settings


def test_secant():
    actual = _secant(x1=-1, y1=2, x2=1, y2=-1)
    expected = 0.33333333333
    assert_allclose(actual, expected)


def test_secant_overflow():
    # https://github.com/nickodell/modab_root_finder/pull/2
    x1 = 0.0
    x2 = 1e15
    x3 = _secant(
        x1=x1,
        y1=-1e+300,
        x2=x2,
        y2=1e+300,
    )
    assert x1 < x3 < x2


@given(
    x1=st.floats(max_value=0),
    y1=st.floats(),
    x2=st.floats(min_value=0),
    y2=st.floats(),
)
@settings(max_examples=1000)
def test_secant_fuzz(x1, y1, x2, y2):
    assume(x1 < x2)
    assume(_sign(y1) != _sign(y2))
    x3 = _secant(x1, y1, x2, y2)
    assert x1 <= x3 <= x2, f"Expected x3 to lie within [x1, x2]. {x1=} {x2=} {x3=}"
