# cython: boundscheck=False, wraparound=False, cdivision=True

# This implementation is based upon the C# code from 
# "modab_modern_impl.cs"

import cython
import os
import math
import modab_root_finder
from libc.math cimport isnan, isinf, NAN, nextafter

cdef bint debug = bool(int(os.environ.get('MODAB_MOD_DEBUG', '0')))


cdef double sign(double x) noexcept:
    # TODO: C# Math.Sign actually can return -1, 0, or 1.
    # Do we want to deal with sign = 0?
    return -1.0 if x < 0.0 else 1.0


cdef double midpoint(double x1, double x2) noexcept:
    # Take the average between p1 X and p2 X
    return (x1 + x2) / 2.0


cdef double secant(double x1, double y1, double x2, double y2) noexcept:
    cdef double x3 = (x1 * y2 - y1 * x2) / (y2 - y1)
    x3 = clamp(x1, x2, x3)
    return x3


cdef double clamp(double x1, double x2, double x3) noexcept:
    # Clamp x3 between x1 and x2. If the function is very flat and y2 is close to
    # y1, floating point rounding errors can shoot x3 outside the bracketing interval

    cdef double x1_shrink = nextafter(x1, x2)
    cdef double x2_shrink = nextafter(x2, x1)
    if x3 < x1_shrink:
        return x1_shrink
    elif x3 > x2_shrink:
        return x2_shrink
    elif isnan(x3):
        return x1
    return x3


cdef show_point_in_context(double x1, double x2, double x3, double y3):
    scaled = (x3 - x1) / (x2 - x1)
    print(f"f({scaled}) = {y3}")


cdef double checked_call(object func, double x):
    if isnan(x) or isinf(x):
        raise modab_root_finder.InternalSolverError()
    cdef double val = func(x)
    if isnan(val) or isinf(val):
        raise modab_root_finder.InvalidSolverInput()
    return val


cpdef modab_modern_impl(F, double x1, double x2, double eps_f, int maxiter=1000):
    cdef double y1, y2, y3
    cdef double ftol = 0.0
    if x1 > x2:
        # Parts of this algorithm assume that x1 < x2.
        # Not sure where.
        x1, x2 = x2, x1
    y1 = checked_call(F, x1)
    y2 = checked_call(F, x2)
    # Are we bisecting right now?
    cdef bint bisection = True
    # What side moved last in AB step?
    cdef int side = 0
    # How long should we try AB before giving up and doing bisection?
    # This setting allows AB to make no progress for 4 iterations before
    # giving up.
    cdef double C = 16
    cdef double x0 = x1
    cdef double ym, r
    if debug:
        print("ModAB Modern Start")
        print("#" * 20)
        print(f"a={x1}, b={x2}")
    for i in range(1, maxiter + 1):
        if bisection:
            x3 = midpoint(x1, x2)
            y3 = checked_call(F, x3)

            if debug:
                show_point_in_context(x1, x2, x3, y3)
            ym = (y1 + y2) / 2.0
            # r is in range [0, 1]
            # symmetry factor
            r = (1 - abs(ym / (y1 - y2)))
            k = r * r
            # Note: if ym and y3 have opposing signs, then this check
            # will always fail.
            if abs(ym - y3) < k * (abs(y3) + abs(ym)):
                if debug:
                    print("switching to false position")

                bisection = False
                # Update threshold for switching back to bisect
                threshold = (x2 - x1) * C
        else:
            x3 = secant(x1, y1, x2, y2)
            y3 = checked_call(F, x3)
            if debug:
                show_point_in_context(x1, x2, x3, y3)
            threshold /= 2.0

        if abs(y3) <= ftol:
            if debug:
                print(f"exiting x converged, {x3=} {y3=}")
            return x3
        x0 = x3  # Keep track of last approximation

        if sign(y1) == sign(y3):
            if side == 1:
                # Apply Anderson Bjork to right side
                # m must be smaller than 1: y3 and y1 have the same
                # sign, so y3 / y1 must be positive.
                # m could be smaller than 0 if p3 was a worse guess
                # than p1.
                m = 1 - y3 / y1
                if m <= 0:
                    y2 /= 2.0
                else:
                    y2 *= m
            if not bisection:
                side = 1
            x1 = x3
            y1 = y3
        else:
            if side == -1:
                # Apply Anderson Bjork to left side
                m = 1 - y3 / y2
                if m <= 0:
                    y1 /= 2.0
                else:
                    y1 *= m
            if not bisection:
                side = -1
            x2 = x3
            y2 = y3
        if abs(x1 - x2) < eps_f:
            # If the bracket p1, p2 is small enough, return
            # success here. Use p3, which is the most recently
            # evaluated point.
            return x3
        if x2 - x1 > threshold:
            if debug and not bisection:
                print("switching back to bisection")
            bisection = True
            side = 0

    raise Exception("failed to converge!")


# Testing wrappers
def _secant(double x1, double y1, double x2, double y2):
    return secant(x1, y1, x2, y2)


def _sign(double x):
    return sign(x)
