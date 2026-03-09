from modab_root_finder import root_scalar
from modab_root_finder.step_introspect import (
    sign,
    SolverState,
    SolverArgs,
    modab_single,
    FuncWrapper,
)
import numpy as np
from numpy.testing import assert_allclose
from modab_root_finder.tests.util import debug_func
import copy


def make_bisect_state(args, x1, x2):
    state = SolverState()
    state.x1 = x1
    state.x2 = x2
    state.y1 = args.func(x1)
    state.y2 = args.func(x2)

    return state


def make_bisect_args(func, **kwargs):
    args = SolverArgs()
    args.func = FuncWrapper(func)
    for name, value in kwargs.items():
        setattr(args, name, value)
    return args


def assert_state_preconditions(state):
    assert state.x1 < state.x2, f"Bracket in wrong order, {state.x1=} {state.x2=}"
    assert sign(state.y1) == -1 * sign(state.y2)
    y2 = state.y2
    assert state.y1 != 0.0
    assert state.y2 != 0.0


def assert_state_postconditions(state, args, prev_state):
    # Check that we made progress
    prev_bracket_size = abs(prev_state.x2 - prev_state.x1)
    bracket_size = abs(state.x2 - state.x1)
    if not state.terminate:
        # assert bracket_size < prev_bracket_size
        # Assert that either the left or right bracket moved
        assert state.x1 > prev_state.x1 or state.x2 < prev_state.x2
    else:
        # We have terminated
        # Check that we were allowed to do that
        if bracket_size <= args.xtol:
            # Terminated due to shrinking bracket
            pass
        elif state.y3 == 0:
            # Terminated due to finding zero
            pass
        else:
            assert False, f"Terminated without justification, {bracket_size=} and y3={state.y3}"
    # This is here to check the validity of our postconditions.
    # Specifically, we need to make sure that our postconditions are not less restrictive than
    # our preconditions in the next loop.
    assert_state_preconditions(state)


def test_bisect1():
    def f(x):
        return x + 1
    args = make_bisect_args(f)
    state = make_bisect_state(args, -5, 5)
    assert_state_preconditions(state)
    # prev_state = state.copy()
    prev_state = copy.copy(state)
    modab_single(state, args)
    assert_state_postconditions(state, args, prev_state)
