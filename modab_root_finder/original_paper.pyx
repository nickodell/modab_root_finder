# This implementation is based upon the C# code from 
# the paper "Modified Anderson-Bjork’s method for
# solving non-linear equations in structural mechanics"

import cython

from scipy.optimize import bisect
import math
import numpy as np
import os


cdef sign(x):
    # TODO: C# Math.Sign actually can return -1, 0, or 1.
    # Do we want to deal with sign = 0?
    return -1 if x < 0 else 1

Precision = np.finfo(float).eps

debug = bool(int(os.environ.get('MODAB_DEBUG', '0')))

@cython.cdivision(True)
cpdef modab_from_paper(F, double x1, double x2, double eps, maxiter=100):
    cdef double y1, y2, y3 = 0
    cdef double x3 = 0
    cdef double ym = 0
    cdef double y1_true = np.nan
    cdef double y2_true = np.nan
    cdef double y3_true = np.nan
    cdef double k = 0.25
    cdef int N = -(math.log2(eps) / 2) + 1
    cdef bool Bisection = True
    cdef int side = 0
    cdef double bracket_min = x1
    cdef double bracket_max = x2
    if debug:
        print(f"a={x1}, b={x2}")
        print(f"{N=}")
    y1 = F(x1); y2 = F(x2)
    y1_true = y1; y2_true = y2
    if debug:
        print(f"f(a)={x1}, f(b)={x2}")
    side = 0
    for i in range(1, maxiter + 1):
        if debug:
            print(f"{i=} {side=}")
        # L17
        if Bisection:
            x3 = (x1 + x2) / 2; y3 = F(x3);  # Midpoint abscissa and function value
            y3_true = y3
            if debug:
                # print(f"f({x3}) = {y3}")
                print(f"f({(x3 - bracket_min) / (bracket_max - bracket_min)}) = {y3}")
            ym = (y1 + y2) / 2;              # Ordinate of chord at midpoint
            ym_true = (y1_true + y2_true) / 2
            if debug:
                # print(f"{y1=} {y1_true=}")
                # print(f"{y2=} {y2_true=}")
                # print(f"{y3=} {y3_true=}")
                # print(f"{ym=} {ym_true=}")
                # print("considering switch to bisect")
                # print(f"{abs(ym - y3)=} < {k * (abs(ym) + abs(y3))=}")
                print(f"linearity check: {abs(ym - y3) / (abs(ym) + abs(y3))=} < {k}")
                # print("before AB correct?")
                # print(f"{abs(ym_true - y3_true)=} < {k * (abs(ym_true) + abs(y3_true))=}")
            if abs(ym - y3) < k * (abs(ym) + abs(y3)):
                if debug:
                    print("Switch to false position")
                Bisection = False           # Switch to false-position
        else: # False position step
            x3 = (x1 * y2 - y1 * x2) / (y2 - y1); y3 = F(x3)
            if debug:
                # print(f"f({x3}) = {y3}")
                print(f"f({(x3 - bracket_min) / (bracket_max - bracket_min)}) = {y3}")
            y3_true = y3
        # L26
        # Note: using 0.5 * eps from GH thread
        if y3 == 0 or abs(x3 - x0) <= 0.5 * eps:   # Convergence check
            return x3;                       # Return the result
        # L28
        x0 = x3;                             # Store the abscissa for the next iteration
        # Apply Anderson Bjork modification
        if side == 1:
            m = 1 - y3 / y1
            if m <= 0:
                # TODO: What does m < 0 represent?
                y2 *= 0.5
            else:
                y2 *= m
            if debug:
                print(f"bjork side=1, m={m} new y2={y2}")
        elif side == 2:
            m = 1 - y3 / y2
            if m <= 0:
                # TODO: What does m < 0 represent?
                y1 *= 0.5
            else:
                y1 *= m
            if debug:
                print(f"bjork side=2, m={m} new y1={y1}")

        # L39
        # print(f"{y1=} {sign(y1)=}")
        if sign(y1) == sign(y3):              # If the left interval does not change sign
            if not Bisection:
                side = 1                     # Store the side that moved
            x1 = x3; y1 = y3;                # Move the left end
            y1_true = y3_true
        else:                                # If the right interval does not change sign
            if not Bisection:
                side = 2                     # Store the side that moved
            x2 = x3; y2 = y3;                # Move the right end
            y2_true = y3_true
        bracket_min = min(x1, x2)
        bracket_max = max(x1, x2)
        if i % N == 0:
            # Is this taking a really long time to converge?
            # If so, re-enable bisection to avoid linear convergence
            # of Anderson-Bjork.
            if debug and not Bisection:
                print("Switching back to bisection")
            Bisection = True


    raise Exception("Gave up, try increasing maxiter.")
