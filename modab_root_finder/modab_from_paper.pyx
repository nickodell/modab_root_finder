import cython

from scipy.optimize import bisect
import math


cdef sign(x):
    return -1 if x < 0 else 1


@cython.cdivision(True)
cpdef modab_from_paper(F, x1, x2, eps):
    cdef float y1, y2, y3
    cdef float x3
    cdef float ym
    cdef int N
    cdef bool Bisection = True
    y1 = F(x1); y2 = F(x2)
    # N = -(Math.Log2(Precision) / 2) + 1;
    N = 100
    side = 0
    for i in range(N):
        # L14
        if Bisection:
            x3 = (x1 + x2) / 2; y3 = F(x3);  # Midpoint abscissa and function value
            ym = (y1 + y2) / 2;              # Ordinate of chord at midpoint
            if abs(ym - y3) < 0.25 * (abs(ym) + abs(y3)):
                # Disable this right now - only bisect
                # Bisection = False           # Switch to false-position
                pass
        else:
            raise NotImplementedError()
        # L24
        print("y3", y3, "y0", F(x0))
        if y3 == 0 or abs(x3 - x0) <= eps:   # Convergence check
            # TODO: Convergence check is done before we know if
            # x3 / x0 is valid bracket
            # print("exit convergence", eps, x3, x0, )
            return x3;                       # Return the result
        # L26
        x0 = x3;                             # Store the abscissa for the next iteration
        # L37
        math_sign_y1 = sign(y1)
        # print(f"{y1=} {sign(y1)=}")
        if sign(y1) == sign(y3):   # If the left interval does not change sign
            if not Bisection: side = 1       # Store the side that move
            x1 = x3; y1 = y3;                # Move the left end
        else:                                # If the right interval does not change sign
            if not Bisection: side = 2       # Store the side that move
            x2 = x3; y2 = y3;                # Move the right end
    return 0
