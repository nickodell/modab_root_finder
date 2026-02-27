import mpmath
import scipy.optimize
import numpy as np
from dataclasses import dataclass


mpmath.mp.dps = 50

@dataclass
class KnownAnswer:
    problem: object
    root: float
    well_behaved: bool
    ambiguity_radius: float
    root_source: str


def use_mpmath_internal(f):
    """Wrap a function. Convert into mpmath on the way in, and float on the way back."""
    def inner(x):
        y = f(mpmath.mpf(x))
        return float(y)
    return inner


# Problems that actually have more than one root and are not unique
multiroot_problem_list = [
    "f42",
    "f83",
    "f84",
    # "f89",
]

def approx_root(f, name, a, b):
    """Find the root of f using bisection."""
    # print("in approx_root")
    # print(f"{name=}, {a=}, {b=}")
    # print(f"{f(a)=}, {f(b)=}")

    # print(f"{a=}, {b=}")
    _, results = scipy.optimize.bisect(f, a, b, xtol=1e-16, rtol=1e-15, full_output=True)

    root = results.root

    if not results.converged:
        raise ValueError("can't find bracket anywhere")

    assert a <= root <= b, f"root {root} not in bracket interval [{a, b}], details {results}"
    return root


def mpmath_root(problem):
    f = problem.f
    a = problem.a
    b = problem.b
    assert a < b
    root_tol = 1e-20
    # print(f"mpmath root for {problem.name=}, {a=}, {b=}")
    root_approx = approx_root(f, problem.name, a, b)
    # print(f"{root_approx=} {f(root_approx)=}")
    f_mpmath = use_mpmath_internal(f)
    root_mpmath = None
    not_zero_at_root = False
    if root_mpmath is None:
        try:
            root_mpmath = mpmath.findroot(f, root_approx, tol=root_tol)
        except ValueError as e:
            if "Could not find root using the given solver" in str(e):
                # This can happen for numerically poorly conditioned functions
                pass
            elif "Could not find root within given tolerance." in str(e):
                # This can happen for numerically poorly conditioned functions
                pass
            else:
                raise
    if root_mpmath is None:
        try:
            root_mpmath = mpmath.findroot(f, [a, b], solver='bisect', tol=root_tol, verify=True)
        except ValueError as e:
            if "Could not find root within given tolerance" in str(e):
                # mpmath has no xtol termination, so discontinuous functions can trigger this
                pass
            else:
                raise
    if root_mpmath is None:
        # Nuclear option. Force the function to have a root near the place where SciPy bisect
        # thinks the root ought to be. mpmath doesn't support xtol, so this is the closest
        # alternative
        not_zero_at_root = True
        center = root_approx
        f_with_root = lambda x: f(x) * ((x - center) ** 2)
        root_mpmath = mpmath.findroot(f_with_root, [a, b], solver='bisect', tol=root_tol)

    root_mpmath = float(root_mpmath)

    # print("mpmath root", root_mpmath)
    # print("f(root) = ", f_mpmath(root_mpmath))
    # print("scipy root", root_approx)
    # print("f(root) = ", f_mpmath(root_approx))
    best_root = None
    # Prefer mpmath's roots - otherwise SciPy bisect gets an advantage
    # simply because it computed the reference value. 
    if abs(f_mpmath(root_mpmath)) + 1e-15 < abs(f_mpmath(root_approx)):
        root = root_mpmath
        root_source = 'scipy'
    else:
        root = root_approx
        root_source = 'mpmath'

    well_behaved = not not_zero_at_root
    if problem.name in multiroot_problem_list:
        well_behaved = False

    return KnownAnswer(
        problem=problem,
        root=root,
        well_behaved=well_behaved,
        ambiguity_radius=find_ambiguity_radius(problem, root),
        root_source=root_source,
    )


def sign(x):
    return -1 if x < 0 else 1


def find_ambiguity_radius(problem, root):
    # print(problem.name, root)
    probe_distance = 1e-15
    iters_since_ambiguity = 0
    increasing = None
    max_ambiguity_radius = 1e-15
    while probe_distance < 0.001:
        if not (problem.a <= root - probe_distance < root + probe_distance <= problem.b):
            # The problem isn't defined here
            break
        a = root - probe_distance
        b = root + probe_distance
        left_val = problem.f(a)
        right_val = problem.f(b)
        # print(f"f({a}) = {left_val} f({b}) = {right_val}")
        if increasing is None and sign(right_val) != sign(left_val):
            if left_val < 0:
                increasing = True
            else:
                increasing = False
        else:
            # We already determined if the function is increasing or decreasing
            if sign(right_val) == sign(left_val):
                # print(f"left and right have same sign {probe_distance=}")
                max_ambiguity_radius = probe_distance
                iters_since_ambiguity = 0
        probe_distance *= 2
        iters_since_ambiguity += 1
        if iters_since_ambiguity > 10:
            # Haven't seen ambiguity in a long time
            break
    return max_ambiguity_radius

