from modab_root_finder import root_scalar
from modab_root_finder.modab_refactor import (
    sign,
    SolverState,
    SolverArgs,
    modab_single,
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


def make_bisect_args(func):
    args = SolverArgs()
    args.func = func
    return args


def assert_state_preconditions(state):
    assert state.x1 < state.x2, "Bracket in wrong order"
    assert sign(state.y1) == -1 * sign(state.y2)


def assert_state_postconditions(state, prev_state):
    # Check that we made progress
    prev_bracket_size = abs(prev_state.x2 - prev_state.x1)
    bracket_size = abs(state.x2 - state.x1)
    assert bracket_size < prev_bracket_size
    # This is here to check the validity of our postconditions.
    # Specifically, we need to make sure that our postconditions are not less restrictive than
    # our preconditions in the next loop.
    assert_state_preconditions(state)


def test_bisect1():
    @debug_func
    def f(x):
        return x + 1
    args = make_bisect_args(f)
    state = make_bisect_state(args, -5, 5)
    assert_state_preconditions(state)
    # prev_state = state.copy()
    prev_state = copy.copy(state)
    print(prev_state.x1)
    modab_single(state, args)
    assert_state_postconditions(state, prev_state)
