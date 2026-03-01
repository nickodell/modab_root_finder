# This implementation is based upon the C# code from 
# "modab_modern_impl.cs"

import cython
import os
import math

debug = bool(int(os.environ.get('MODAB_MOD_DEBUG', '0')))


cdef double sign(double x):
    # TODO: C# Math.Sign actually can return -1, 0, or 1.
    # Do we want to deal with sign = 0?
    return -1.0 if x < 0.0 else 1.0


@cython.final
cdef class SolverState:
    pass


@cython.final
cdef class Node:
    cdef double x
    cdef double y
    def __str__(self):
        return f"N({self.x=}, {self.y=})"
    def __repr__(self):
        return f"N({self.x=}, {self.y=})"


cdef double midpoint(Node p1, Node p2):
    # Take the average between p1 X and p2 X
    return (p1.x + p2.x) / 2.0


cdef double secant(Node p1, Node p2):
    return (p1.x * p2.y - p1.y * p2.x) / (p2.y - p1.y)


cdef tuple initialize(F, double x1, double x2, double eps_f):
    p1 = Node()
    p1.x = x1
    p1.y = F(x1)
    p2 = Node()
    p2.x = x2
    p2.y = F(x2)
    if sign(p1.y) == sign(p2.y):
        raise Exception("bad starting bracket")
    eps = Node()
    # Note: deviate from Solver.cs in two ways.
    # First, precision here is absolute, not relative
    # Second, termination due to ftol is disabled
    eps.x = eps_f
    eps.y = 0  # eps_f / 100
    return p1, p2, eps


cdef show_point_in_context(Node p1, Node p2, Node p3):
    scaled = (p3.x - p1.x) / (p2.x - p1.x)
    print(f"f({scaled}) = {p3.y}")


@cython.cdivision(True)
cpdef modab_modern_impl(F, double x1, double x2, double eps_f, int maxiter=1000):
    cdef Node p1, p2, eps
    if x1 > x2:
        # Parts of this algorithm assume that x1 < x2.
        # Not sure where.
        x1, x2 = x2, x1
    p1, p2, eps = initialize(F, x1, x2, eps_f)
    # Are we bisecting right now?
    cdef bint bisection = True
    # What side moved last in AB step?
    cdef int side = 0
    # How long should we try AB before giving up and doing bisection?
    # This setting allows AB to make no progress for 4 iterations before
    # giving up.
    cdef double C = 16
    cdef double x0 = p1.x
    cdef double ym, r
    cdef Node p3
    if debug:
        print("ModAB Modern Start")
        print("#" * 20)
        print(f"a={x1}, b={x2}")
    for i in range(1, maxiter + 1):
        if debug:
            print(f"\n" * 3)
        if bisection:
            p3 = Node()
            # if debug:
            #     print(f"{p1=} {p2=}")
            p3.x = midpoint(p1, p2)
            p3.y = F(p3.x)

            if debug:
                show_point_in_context(p1, p2, p3)
            ym = (p1.y + p2.y) / 2.0
            # r is in range [0, 1]
            # symmetry factor
            r = (1 - abs(ym / (p1.y - p2.y)))
            k = r * r
            if debug:
                pass
                # Note: if ym and p3.y have opposing signs, then this check
                # will always fail.
                print(f"{k=}")
                print(f"expected y: {ym}")
                print(f"actual y: {p3.y}")
                print(f"check: {abs(ym - p3.y) < k * (abs(p3.y) + abs(ym))}")
                print(f"num: {abs(ym - p3.y)}")
                print(f"denom: {(abs(p3.y) + abs(ym))}")
                print(f"ratio: {abs(ym - p3.y) / (abs(p3.y) + abs(ym))} < {k}")
                # print(f"{p1.y=} {p2.y=} {ym}")
                # print(f"{p3.y=} {ym=}")
                # print(f"{r=}")
                # print(f"{k=}")
                # print(f"check: {abs(ym - p3.y)=} < {k * (abs(p3.y) + abs(ym))=}")
            if abs(ym - p3.y) < k * (abs(p3.y) + abs(ym)):
            # if debug:
            #     print(f"expected y: {ym}")
            #     print(f"actual y: {p3.y}")
            #     print(f"check: {abs(ym - p3.y) < k * abs(p1.y - p2.y)}")
            #     print(f"ratio: {abs(ym - p3.y) / abs(p1.y - p2.y)} < {k}")
            # if abs(ym - p3.y) < k * abs(p1.y - p2.y):

                if debug:
                    print("switching to false position")

                bisection = False
                # Update threshold for switching back to bisect
                threshold = (p2.x - p1.x) * C
        else:
            p3 = Node()
            p3.x = secant(p1, p2)
            p3.y = F(p3.x)
            if debug:
                show_point_in_context(p1, p2, p3)
            threshold /= 2.0

        if debug:
            print(f"{abs(p3.y) <= eps.y=} or {abs(p3.x - x0) <= eps.x=}")
            print(f"{abs(p3.y)=} <= {eps.y=} or {abs(p3.x - x0)=} <= {eps.x=}")
        if abs(p3.y) <= eps.y or abs(p3.x - x0) <= eps.x:
            if debug:
                print(f"exiting x converged, {p3}")
            return p3.x
        x0 = p3.x  # Keep track of last approximation

        if sign(p1.y) == sign(p3.y):
            if side == 1:
                # Apply Anderson Bjork to right side
                # m must be smaller than 1: p3.y and p1.y have the same
                # sign, so p3.y / p1.y must be positive.
                # m could be smaller than 0 if p3 was a worse guess
                # than p1.
                m = 1 - p3.y / p1.y
                if m <= 0:
                    p2.y /= 2.0
                else:
                    p2.y *= m
            if not bisection:
                side = 1
            p1 = p3
        else:
            if side == -1:
                # Apply Anderson Bjork to left side
                m = 1 - p3.y / p2.y
                if m <= 0:
                    p1.y /= 2.0
                else:
                    p1.y *= m
            if not bisection:
                side = -1
            p2 = p3
        if abs(p1.x - p2.x) < eps.x:
            # If the bracket p1, p2 is small enough, return
            # success here. Use p3, which is the most recently
            # evaluated point.
            return p3.x
        # bug here? Can p2.x - p1.x be negative?
        # Yes, if bracket was specified that way at beginning.
        if abs(p2.x - p1.x) > threshold:
            if debug and not bisection:
                print("switching back to bisection")
            bisection = True
            side = 0

    raise Exception("failed to converge!")
