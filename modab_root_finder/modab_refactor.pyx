# cython: boundscheck=False, wraparound=False, cdivision=True

import cython
import os
from libc.math cimport isnan, NAN


cdef bint debug = bool(int(os.environ.get('MODAB_REFACTOR_DEBUG', '0')))
cdef bint enable_prev_x_check = False


cpdef double sign(double x):
    return -1.0 if x < 0.0 else 1.0


cpdef double midpoint(double x1, double x2):
    return (x1 + x2) / 2.0


@cython.final
cdef class SolverArgs:
    cdef double xtol
    cdef double rtol
    cdef double ftol
    cdef long maxiter
    cdef object func

    def __init__(self):
        self.xtol = 0
        self.rtol = 0
        self.ftol = 0
        self.func = None

    @property
    def xtol(self):
        return self.xtol

    @xtol.setter
    def xtol(self, value):
        self.xtol = value

    # rtol
    # ftol

    @property
    def func(self):
        return self.func

    @func.setter
    def func(self, value):
        self.func = value


@cython.final
cdef class SolverState:
    cdef bint bisect
    # The ends of the bracket. x1 < x2
    cdef double x1, x2
    # New point
    cdef double x3
    # The value of x1 and x2, modified by AB corrections
    cdef double y1, y2
    cdef double y3
    cdef long func_calls
    # The value of x3 in the previous iteration
    cdef double x3_prev
    cdef bint terminate
    cdef double threshold
    cdef int side

    def __init__(self):
        self.bisect = True
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.func_calls = 0
        self.x3_prev = 0
        self.terminate = False
        self.threshold = 0
        self.side = 0

    @property
    def terminate(self):
        return self.terminate

    @terminate.setter
    def terminate(self, value):
        self.terminate = value

    @property
    def x1(self):
        return self.x1

    @x1.setter
    def x1(self, value):
        self.x1 = value

    @property
    def x2(self):
        return self.x2

    @x2.setter
    def x2(self, value):
        self.x2 = value

    @property
    def y1(self):
        return self.y1

    @y1.setter
    def y1(self, value):
        self.y1 = value

    @property
    def y2(self):
        return self.y2

    @y2.setter
    def y2(self, value):
        self.y2 = value

    @property
    def x3_prev(self):
        return self.x3_prev

    @x3_prev.setter
    def x3_prev(self, value):
        self.x3_prev = value

    @property
    def terminate(self):
        return self.terminate

    @terminate.setter
    def terminate(self, value):
        self.terminate = value

    @property
    def threshold(self):
        return self.threshold

    @threshold.setter
    def threshold(self, value):
        self.threshold = value

    @property
    def x3(self):
        return self.x3

    @x3.setter
    def x3(self, value):
        self.x3 = value

    @property
    def side(self):
        return self.side

    @side.setter
    def side(self, value):
        self.side = value


cpdef tuple initialize(object func, double x1, double x2, double eps_f, int maxiter):
    state = SolverState()
    args = SolverArgs()
    args.xtol = eps_f
    args.rtol = 0
    args.ftol = 0
    args.func = func
    args.maxiter = maxiter

    if x1 > x2:
        # x1 must be lower bracket for Anderson Bjork correction to work.
        x1, x2 = x2, x1

    state.x1 = x1
    state.x2 = x2
    state.y1 = func(x1)
    state.y2 = func(x2)
    state.func_calls = 2
    return state, args


cpdef double modab_single(SolverState state, SolverArgs args):
    # How long should we try AB before giving up and doing bisection?
    # This setting allows AB to make no progress for 4 iterations before
    # giving up.
    cdef double C = 16
    cdef double ym, r
    if debug:
        print(f"\n" * 3)
    if state.bisect:
        if debug:
            print(f"x1={state.x1} x2={state.x2}")
            print(f"f(x1)={state.y1} f(x2)={state.y2}")
        state.x3 = (state.x1 + state.x2) / 2.0
        state.y3 = args.func(state.x3)

        ym = (state.y1 + state.y2) / 2.0
        # r is in range [0, 1]
        # symmetry factor
        r = (1 - abs(ym / (state.y1 - state.y2)))
        k = r * r
        if debug:
            pass
            # Note: if ym and state.y3 have opposing signs, then this check
            # will always fail.
            print(f"{k=}")
            print(f"expected y: {ym}")
            print(f"actual y: {state.y3}")
            print(f"check: {abs(ym - state.y3) < k * (abs(state.y3) + abs(ym))}")
            print(f"num: {abs(ym - state.y3)}")
            print(f"denom: {(abs(state.y3) + abs(ym))}")
            print(f"ratio: {abs(ym - state.y3) / (abs(state.y3) + abs(ym))} < {k}")
            # print(f"{state.x1=} {state.x2=} {ym}")
            # print(f"{state.y3=} {ym=}")
            # print(f"{r=}")
            # print(f"{k=}")
            # print(f"check: {abs(ym - state.y3)=} < {k * (abs(state.y3) + abs(ym))=}")
        if abs(ym - state.y3) < k * (abs(state.y3) + abs(ym)):
        # if debug:
        #     print(f"expected y: {ym}")
        #     print(f"actual y: {state.y3}")
        #     print(f"check: {abs(ym - state.y3) < k * abs(state.x1 - state.x2)}")
        #     print(f"ratio: {abs(ym - state.y3) / abs(state.x1 - state.x2)} < {k}")
        # if abs(ym - state.y3) < k * abs(state.x1 - state.x2):

            if debug:
                print("switching to false position")

            state.bisect = False
            # Update threshold for switching back to bisect
            state.threshold = (state.x2 - state.x1) * C
    else:
        state.x3 = (state.x1 * state.y2 - state.y1 * state.x2) / (state.y2 - state.y1)
        state.y3 = args.func(state.x3)
        state.threshold /= 2.0

    if debug:
        print(f"{abs(state.y3) <= args.xtol=} or {abs(state.x3 - state.x3_prev) <= args.xtol=}")
        print(f"{abs(state.y3)=} <= {args.ftol=} or {abs(state.x3 - state.x3_prev)=} <= {args.xtol=}")
    if abs(state.y3) <= args.ftol or (abs(state.x3 - state.x3_prev) <= args.xtol and enable_prev_x_check):
        state.terminate = True
        return state.x3
    state.x3_prev = state.x3  # Keep track of last approximation

    if sign(state.y1) == sign(state.y3):
        if state.side == 1:
            # Apply Anderson Bjork to right side
            # m must be smaller than 1: state.y3 and state.y1 have the same
            # sign, so state.y3 / state.y1 must be positive.
            # m could be smaller than 0 if p3 was a worse guess
            # than p1.
            m = 1 - state.y3 / state.y1
            if m <= 0:
                state.y2 /= 2.0
            else:
                state.y2 *= m
        if not state.bisect:
            state.side = 1
        # p1 = p3
        state.x1 = state.x3
        state.y1 = state.y3
    else:
        if state.side == -1:
            # Apply Anderson Bjork to left side
            m = 1 - state.y3 / state.y2
            if m <= 0:
                state.y1 /= 2.0
            else:
                state.y1 *= m
        if not state.bisect:
            state.side = -1
        # p2 = p3
        state.x2 = state.x3
        state.y2 = state.y3
    if debug:
        print("bracket size", abs(state.x1 - state.x2), "xtol", args.xtol)
    if abs(state.x1 - state.x2) < args.xtol:
        # If the bracket p1, p2 is small enough, return
        # success here. Use p3, which is the most recently
        # evaluated point.
        state.terminate = True
        return state.x3
    if state.x2 - state.x1 > state.threshold:
        if debug and not state.bisect:
            print("switching back to bisection")
        state.bisect = True
        state.side = 0
    return NAN


def modab_refactor(object func, double x1, double x2, double xtol, int maxiter=1000):
    cdef SolverState state
    cdef SolverArgs args
    state, args = initialize(func, x1, x2, xtol, maxiter)
    for _ in range(maxiter):
        root = modab_single(state, args)
        if state.terminate:
            return root
    raise Exception("failed to converge!")
