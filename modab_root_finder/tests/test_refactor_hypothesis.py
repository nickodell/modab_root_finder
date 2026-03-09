from hypothesis import given, strategies as st, assume, settings, example
import copy
import numpy as np
from modab_root_finder.tests.test_refactor import (
    make_bisect_state,
    make_bisect_args,
    assert_state_preconditions,
    assert_state_postconditions,
)
from modab_root_finder.tests.util import (
    debug_func
)
from modab_root_finder.step_introspect import (
    modab_single,
    sign,
)
from modab_root_finder import (
    InvalidSolverInput,
)


min_nonzero = 5e-324
eps = np.finfo(float).eps
min_rtol = eps * 4


@given(
    x1=st.floats(max_value=0),
    x2=st.floats(min_value=0),
    side=st.sampled_from([-1, 0, 1]),
    bisect=st.booleans(),
    # Note: supplying a value near 0 for y1 or y2 can result in underflow,
    # which changes the sign of y1 or y2. TODO: figure out if this is a bug.
    y1=st.floats(max_value=-eps, allow_infinity=False),
    y2=st.floats(min_value=eps, allow_infinity=False),
    y3=st.floats(),
    xtol=st.sampled_from([1e-14, 1e-10, 1e-8, 1]),
)
@settings(max_examples=1000)
def test_single_step(x1, x2, side, bisect, y1, y2, y3, xtol):
    assume(x1 < x2)
    assume((x2 - x1) > max(abs(x1), abs(x2)) * min_rtol + xtol)
    # Require that our brackets have opposite signs
    if sign(y1) == sign(y2):
        y2 = -y2
    assume(sign(y1) != sign(y2))
    # Require that our brackets are nonzero
    # Otherwise we'd have terminated last iteration
    assume(y1 != 0)
    assume(y2 != 0)
    def f(x):
        vals = {
            x1: y1,
            x2: y2,
        }
        if x in vals:
            return vals[x]
        if not (x1 < x < x2):
            raise ValueError(f"evaluated function outside range! {x=}")
        # assume any value inside range results in y3
        return y3
    args = make_bisect_args(f, xtol=xtol)
    try:
        state = make_bisect_state(args, x1, x2)
    except InvalidSolverInput:
        assume(False)
    state.side = side
    state.bisect = bisect
    assert_state_preconditions(state)
    prev_state = copy.copy(state)
    try:
        modab_single(state, args)
    except InvalidSolverInput:
        assume(False)
    assert_state_postconditions(state, args, prev_state)
