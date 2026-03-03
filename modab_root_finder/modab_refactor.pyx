# cython: boundscheck=False, wraparound=False, cdivision=True

import cython
from libc.math cimport isnan, NAN


cpdef double sign(double x):
    return -1.0 if x < 0.0 else 1.0


cpdef double midpoint(double x1, double x2):
    return (x1 + x2) / 2.0


@cython.final
cdef class SolverArgs:
    cdef double xtol
    cdef double rtol
    cdef double ftol
    cdef object func

    @property
    def func(self):
        return self.func

    def __init__(self, asdict):
        self.xtol = asdict['xtol']
        self.rtol = asdict['rtol']
        self.ftol = asdict['ftol']
        self.func = asdict['func']

    def asdict(self):
        return {
            'xtol': self.xtol,
            'rtol': self.rtol,
            'ftol': self.ftol,
            'func': self.func,
        }



@cython.final
cdef class SolverState:
    cdef bint bisect
    # The ends of the bracket. x1 < x2
    cdef double x1, x2
    # The value of x1 and x2, modified by AB corrections
    cdef double y1, y2
    cdef long func_calls
    # The value of x3 in the previous iteration
    cdef double x3_prev
    cdef bint terminate

    def __init__(self, asdict):
        self.bisect = asdict['bisect']
        self.x1 = asdict['x1']
        self.x2 = asdict['x2']
        self.y1 = asdict['y1']
        self.y2 = asdict['y2']
        self.func_calls = asdict['func_calls']
        self.x3_prev = asdict['x3_prev']
        self.terminate = asdict['terminate']

    def asdict(self):
        return {
            'bisect': self.bisect,
            'x1': self.x1,
            'x2': self.x2,
            'y1': self.y1,
            'y2': self.y2,
            'func_calls': self.func_calls,
            'x3_prev': self.x3_prev,
            'terminate': self.terminate,
        }


cpdef double modab_single(SolverState state, SolverArgs args):
    cdef double x3 = 0, tol = 0, y3 = 0
    tol = args.rtol + state.x1 * args.rtol
    if state.bisect:
        # Find x3
        x3 = midpoint(state.x1, state.x2)
        if abs(x3 - state.x1) < tol:
            return x3
        state.func_calls += 1
        y3 = args.func(x3)
        if y3 <= args.ftol:
            return x3
    else:
        pass
    state.x3_prev = x3
    return NAN

    # find_next_point()


def modab_refactor(a, b, c, d, e):
    pass
