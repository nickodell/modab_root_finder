# Note: copied from https://github.com/Proektsoftbg/Numerical/tree/main/Numerical-SciPy
# Copyright Ned Ganchovski
# License: MIT License

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import bisect as sp_bisect
from scipy.optimize import brenth, brentq, elementwise
from scipy.optimize import ridder as sp_ridder
from scipy.optimize import toms748 as sp_toms748
import matplotlib.pyplot as plt

from modab_root_finder import (
    modab_from_paper,
    modab_from_proektsoftbg,
    mpmath_root,
)


def mod_ab_author(f, left, right, target, precision=1e-14):
    g = (lambda x: f(x) - target) if target != 0 else f
    #return modab_from_paper(g, left, right, precision)
    return modab_from_proektsoftbg(g, left, right, precision * 0.5)


# Function-call counting wrapper
class CountedFunc:
    """Wraps a callable and counts the number of evaluations."""
    __slots__ = ('_f', 'count')

    def __init__(self, f):
        self._f = f
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self._f(x)

    def reset(self):
        self.count = 0

# Scipy solver wrappers
_MIN_RTOL = 4 * np.finfo(float).eps  # ~8.88e-16, scipy's minimum

def make_scipy_solver(scipy_func, name):
    """Create a wrapper that matches the (f, left, right, target, precision) signature."""
    def solver(f, left, right, target, precision=1e-14):
        g = (lambda x: f(x) - target) if target != 0 else f
        a, b = min(left, right), max(left, right)
        try:
            # print("func", scipy_func, "xtol", precision, "rtol", _MIN_RTOL)
            root, info = scipy_func(g, a, b, xtol=precision, rtol=_MIN_RTOL,
                                    maxiter=1000, full_output=True, disp=False)
            # print(info)
            return root
        except (ValueError, RuntimeError):
            return float('nan')
    solver.__name__ = name
    return solver

def wrap_find_root():
    """Create a wrapper for find_root that matches the required signature."""
    def solver(f, left, right, target, precision=1e-14):
        # print("wrap_find_root 1", f.count)
        g = (np.vectorize((lambda x: f(x) - target), otypes=[np.float64]) 
             if target != 0 else np.vectorize(f, otypes=[np.float64]))
        # print("wrap_find_root 2", f.count)
        # g = (lambda x: np.array([f(x.item()) - target]))
        a, b = min(left, right), max(left, right)
        # print("wrap_find_root 3", f.count)
        try:
            tolerances = dict(xatol=precision, xrtol=0, fatol=0, frtol=0)
            # print("wrap_find_root 4", f.count)
            # print(g)
            res = elementwise.find_root(g, (a, b), tolerances=tolerances)
            # print("wrap_find_root 5", f.count)
            return res.x
        except (ValueError, RuntimeError):
            return float('nan')
    solver.__name__ = 'sp_chandrupatla'
    return solver

scipy_bisect = make_scipy_solver(sp_bisect, "sp_bisect")
scipy_brentq = make_scipy_solver(brentq,    "sp_brentq")
scipy_brenth = make_scipy_solver(brenth,    "sp_brenth")
scipy_ridder = make_scipy_solver(sp_ridder, "sp_ridder")
scipy_toms748 = make_scipy_solver(sp_toms748, "sp_toms748")
scipy_chandrupatla = wrap_find_root()

# Problem definition
@dataclass
class Problem:
    name: str
    f: Callable[[float], float]
    a: float
    b: float
    value: float = 0.0


def P(x):
    return x + 1.11111

# Test problems
problems1 = [
    # Sérgio Galdino. A family of regula falsi root-finding methods
    Problem("f01", lambda x: x**3 - 1, 0.5, 1.5),
    Problem("f02", lambda x: x**2 * (x**2 / 3 + math.sqrt(2) * math.sin(x)) - math.sqrt(3) / 18, 0.1, 1),
    Problem("f03", lambda x: 11 * x**11 - 1, 0.1, 1),
    Problem("f04", lambda x: x**3 + 1, -1.8, 0),
    Problem("f05", lambda x: x**3 - 2 * x - 5, 2, 3),
    Problem("f06", lambda x: 2 * x * math.exp(-5) + 1 - 2 * math.exp(-5 * x), 0, 1),
    Problem("f07", lambda x: 2 * x * math.exp(-10) + 1 - 2 * math.exp(-10 * x), 0, 1),
    Problem("f08", lambda x: 2 * x * math.exp(-20) + 1 - 2 * math.exp(-20 * x), 0, 1),
    Problem("f09", lambda x: (1 + (1 - 5)**2) * x**2 - (1 - 5 * x)**2, 0, 1),
    Problem("f10", lambda x: (1 + (1 - 10)**2) * x**2 - (1 - 10 * x)**2, 0, 1),
    Problem("f11", lambda x: (1 + (1 - 20)**2) * x**2 - (1 - 20 * x)**2, 0, 1),
    Problem("f12", lambda x: x**2 - (1 - x)**5, 0, 1),
    Problem("f13", lambda x: x**2 - (1 - x)**10, 0, 1),
    Problem("f14", lambda x: x**2 - (1 - x)**20, 0, 1),
    Problem("f15", lambda x: (1 + (1 - 5)**4) * x - (1 - 5 * x)**4, 0, 1),
    Problem("f16", lambda x: (1 + (1 - 10)**4) * x - (1 - 10 * x)**4, 0, 1),
    Problem("f17", lambda x: (1 + (1 - 20)**4) * x - (1 - 20 * x)**4, 0, 1),
    Problem("f18", lambda x: math.exp(-5 * x) * (x - 1) + x**5, 0, 1),
    Problem("f19", lambda x: math.exp(-10 * x) * (x - 1) + x**10, 0, 1),
    Problem("f20", lambda x: math.exp(-20 * x) * (x - 1) + x**20, 0, 1),
    Problem("f21", lambda x: x**2 + math.sin(x / 5) - 1 / 4, 0, 1),
    Problem("f22", lambda x: x**2 + math.sin(x / 10) - 1 / 4, 0, 1),
    Problem("f23", lambda x: x**2 + math.sin(x / 20) - 1 / 4, 0, 1),
    Problem("f24", lambda x: (x + 2) * (x + 1) * (x - 3)**3, 2.6, 4.6),
    Problem("f25", lambda x: (x - 4)**5 * math.log(x), 3.6, 5.6),
    Problem("f26", lambda x: (math.sin(x) - x / 4)**3, 2, 4),
    Problem("f27", lambda x: (81 - P(x) * (108 - P(x) * (54 - P(x) * (12 - P(x))))) * (1 if P(x) < 3 else (-1 if P(x) > 3 else 0)), 1, 3),
    Problem("f28", lambda x: math.sin((x - 7.143)**3), 7, 8),
    Problem("f29", lambda x: math.exp((x - 3)**5) - 1, 2.6, 4.6),
    Problem("f30", lambda x: math.exp((x - 3)**5) - math.exp(x - 1), 4, 5),
    Problem("f31", lambda x: math.pi - 1 / x, 0.05, 5),
    Problem("f32", lambda x: 4 - math.tan(x), 0, 1.5),
    Problem("f33", lambda x: math.cos(x) - x**3, 0, 4),
    # Steven A. Stage. Comments on An Improvement to the Brent’s Method 
    Problem("f34", lambda x: math.cos(x) - x, -11, 9),
    Problem("f35", lambda x: math.sqrt(abs(x - 2 / 3)) * (1 if x <= 2 / 3 else -1) - 0.1, -11, 9),
    Problem("f36", lambda x: abs(x - 2 / 3)**0.2 * (1 if x <= 2 / 3 else -1), -11, 9),
    Problem("f37", lambda x: (x - 7 / 9)**3 + (x - 7 / 9) * 1e-3, -11, 9),
    Problem("f38", lambda x: -0.5 if x <= 1 / 3 else 0.5, -11, 9),
    Problem("f39", lambda x: -1e-3 if x <= 1 / 3 else 1 - 1e-3, -11, 9),
    # Note: discontinuous root
    Problem("f40", lambda x: 0 if (x - 2 / 3) == 0 else 1 / (x - 2 / 3), -11, 9),
    # A. Swift and G.R. Lindfield. Comparison of a Continuation Method with Brents Method for the Numerical Solution of a Single Nonlinear Equation
    Problem("f41", lambda x: 2 * x * math.exp(-5) - 2 * math.exp(-5 * x) + 1, 0, 10),
    Problem("f42", lambda x: (x**2 - x - 6) * (x**2 - 3 * x + 2), 0, math.pi),
    Problem("f43", lambda x: x**3, -1, 1.5),
    Problem("f44", lambda x: x**5, -1, 1.5),
    Problem("f45", lambda x: x**7, -1, 1.5),
    # Problem("f45_rescale", lambda x: x**7*1e5, -1, 1.5),
    Problem("f46", lambda x: (math.exp(-5 * x) - x - 0.5) / x**5, 0.09, 0.7),
    Problem("f47", lambda x: 1 / math.sqrt(x) - 2 * math.log(5e3 * math.sqrt(x)) + 0.8, 0.0005, 0.5),
    Problem("f48", lambda x: 1 / math.sqrt(x) - 2 * math.log(5e7 * math.sqrt(x)) + 0.8, 0.0005, 0.5),
    Problem("f49", lambda x: (-x**3 - x - 1) if x <= 0 else (x**(1 / 3) - x - 1), -1, 1),
    Problem("f50", lambda x: x**3 - 2 * x - x + 3, -3, 2),
    Problem("f51", lambda x: math.log(x), 0.5, 5),
    Problem("f52", lambda x: (10 - x) * math.exp(-10 * x) - x**10 + 1, 0.5, 8),
    Problem("f53", lambda x: math.exp(math.sin(x)) - x - 1, 1.0, 4),
    Problem("f54", lambda x: 2 * math.sin(x) - 1, 0.1, math.pi / 3),
    Problem("f55", lambda x: (x - 1) * math.exp(-x), 0.0, 1.5),
    Problem("f56", lambda x: (x - 1)**3 - 1, 1.5, 3),
    Problem("f57", lambda x: math.exp(x**2 + 7 * x - 30) - 1, 2.6, 3.5),
    Problem("f58", lambda x: math.atan(x) - 1, 1.0, 8),
    Problem("f59", lambda x: math.exp(x) - 2 * x - 1, 0.2, 3),
    Problem("f60", lambda x: math.exp(-x) - x - math.sin(x), 0.0, 2),
    Problem("f61", lambda x: x**2 - math.sin(x)**2 - 1, -1, 2),
    Problem("f62", lambda x: math.sin(x) - x / 2, math.pi / 2, math.pi),
]
# Oliveira I. F. D., Takahashi R. H. C.
# An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality
problems2 = [
    Problem("f63", lambda x: x * math.exp(x) - 1, -1, 1),
    Problem("f64", lambda x: math.tan(x - 1 / 10), -1, 1),
    Problem("f65", lambda x: math.sin(x) + 0.5, -1, 1),
    Problem("f66", lambda x: 4 * x**5 + x * x + 1, -1, 1),
    Problem("f67", lambda x: x + x**10 - 1, -1, 1),
    Problem("f68", lambda x: math.pi**x - math.e, -1, 1),
    Problem("f69", lambda x: math.log(abs(x - 10 / 9)), -1, 1),
    Problem("f70", lambda x: 1 / 3 + (1 if x > 0 else (-1 if x < 0 else 0)) * abs(x)**(1 / 3) + x**3, -1, 1),
    Problem("f71", lambda x: (x + 2 / 3) / (x + 101 / 100), -1, 1),
    Problem("f72", lambda x: (x * 1e6 - 1)**3, -1, 1),
    Problem("f73", lambda x: math.exp(x) * (x * 1e6 - 1)**3, -1, 1),
    Problem("f74", lambda x: (x - 1 / 3)**2 * math.atan(x - 1 / 3), -1, 1),
    Problem("f75", lambda x: (1 if 3 * x - 1 > 0 else (-1 if 3 * x - 1 < 0 else 0)) * (1 - math.sqrt(1 - (3 * x - 1)**2 / 81)), -1, 1),
    Problem("f76", lambda x: (1 + 1e6) / 1e6 if x > (1 - 1e6) / 1e6 else -1, -1, 1),
    # Note: has no root
    Problem("f77", lambda x: 1 / (21 * x - 1) if x != 1 / 21 else 0, -1, 1),
    Problem("f78", lambda x: x * x / 4 + math.ceil(x / 2) - 0.5, -1, 1),
    Problem("f79", lambda x: math.ceil(10 * x - 1) + 0.5, -1, 1),
    Problem("f80", lambda x: x + math.sin(x * 1e6) / 10 + 1e-3, -1, 1),
    Problem("f81", lambda x: 1 + math.sin(1 / (x + 1)) - 1e-15 if x > -1 else -1, -1, 1),
    Problem("f82", lambda x: 202 * x - 2 * math.floor((2 * x + 1e-2) / 2e-2) - 0.1, -1, 1),
    Problem("f83", lambda x: (202 * x - 2 * math.floor((2 * x + 1e-2) / 2e-2) - 0.1)**3, -1, 1),
]
# SciML project benchmarks suite
problems3 = [
    Problem("f84", lambda x: (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) - 0.05, 0.5, 5.5),
    Problem("f85", lambda x: math.sin(x) - 0.5 * x - 0.3, -10.0, 10.0),
    Problem("f86", lambda x: math.exp(x) - 1 - x - x * x / 2 - 0.005, -2.0, 2.0),
    Problem("f87", lambda x: 1 / (x - 0.5) - 2 - 0.05, 0.6, 2.0),
    Problem("f88", lambda x: math.log(x) - x + 2 - 0.05, 0.1, 3.0),
    Problem("f89", lambda x: math.sin(20 * x) + 0.1 * x - 0.1, -4.0, 5.0),
    Problem("f90", lambda x: x**3 - 2 * x**2 + x - 0.025, -1.0, 2.0),
    Problem("f91", lambda x: x * math.sin(1 / x) - 0.1 - 0.01, 0.01, 1.0),
]

all_problems = problems1 + problems2 + problems3

problem_lookup = {p.name: p for p in all_problems}

# Solver table
solvers = [
    ("bisect", scipy_bisect),
    ("brentq", scipy_brentq),
    ("brenth", scipy_brenth),
    ("ridder", scipy_ridder),
    ("chandr", scipy_chandrupatla),
    ("  toms", scipy_toms748),
    (" modAB", mod_ab_author),
    # (" paper", mod_ab_from_paper),
]


def get_true_answer(p):
    assert p.value == 0
    known_answer = mpmath_root(p)
    return known_answer


true_answers = {}


def init_true_answers():
    for p in all_problems:
        true_answers[p.name] = get_true_answer(p)


def sign(x):
    return -1 if x < 0 else 1


def find_nearby_root(f, x0):
    search_area = 1e-10
    for i in range(10):
        a = x0 - search_area
        b = x0 + search_area
        if sign(f(a)) != sign(f(b)):
            _, results = sp_bisect(f, a, b, full_output=True, xtol=1e-15)
            return results
        search_area *= 2
    raise Exception()



def check_solution(f, p, root, eps):
    true_answer = true_answers[p.name]
    # Note: some of the benchmark problems have a large area where
    # f(x) == 0. Add the 'ambiguity radius' to eps to make those
    # cases more lenient
    if abs(root - true_answer.root) < eps + true_answer.ambiguity_radius:
        return
    if not true_answer.well_behaved:
        return
    froot = f(root)
    if froot <= eps:
        # This root is small enough
        return

    results = find_nearby_root(f, root)
    if results.converged:
        if f(results.root) < eps:
            # How different is it from our root?
            x_err = abs(results.root - root)
            print(f"solver's root off, {x_err=}, f({root}) = {froot}")
            if x_err > eps:
                raise Exception("Found bad solution")
            else:
                return
    else:
        raise Exception("bisection didn't converge")


def termsearch(args):
    # eps_vals = [1e-14, 1e-12, 1e-10, 1e-8]
    eps_vals = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
    for problem in all_problems:
        for eps in eps_vals:
            for solver_name, solver in solvers:
                print("solver", solver_name, "problem", problem.name, "eps", eps)
                cf = CountedFunc(problem.f)
                root = solver(cf, problem.a, problem.b, problem.value, eps)
                check_solution(problem.f, problem, root, eps)


def showsolutions(args):
    global all_problems
    if args.func is not None:
        # filter function list
        all_problems = [p for p in all_problems if p.name == args.func]
    init_true_answers()    
    df_rows = []
    for problem in all_problems:
        problem_name = problem.name
        answer = true_answers[problem_name]
        digits_right = int(-math.log10(max(abs(problem.f(answer.root)), 1e-15)))
        df_rows.append({
            'name': problem_name,
            'root': answer.root,
            'f(x)': problem.f(answer.root),
            'amb': answer.ambiguity_radius,
            'digits': digits_right,
            'wb': answer.well_behaved,
            'source': answer.root_source,
        })
    df = pd.DataFrame(df_rows)
    print(df.to_string())
    print("done")



# Benchmark runner
def bench(args):
    global all_problems
    if args.func is not None:
        # filter function list
        all_problems = [p for p in all_problems if p.name == args.func]

    sections = args.sections.split(",")

    enable_roots = False
    enable_acc = False
    enable_nfev = False
    enable_fval = False

    if "roots" in sections:
        enable_roots = True
    if "acc" in sections:
        enable_acc = True
    if "nfev" in sections:
        enable_nfev = True
    if "fval" in sections:
        enable_fval = True

    eps = 1e-10
    col_w = 22  # column width for results

    if enable_roots:
        # Results
        print("Roots found")
        header = f"{'Func':>4}; " + "; ".join(f"{name:>{col_w}}" for name, _ in solvers)
        print(header)
        for p in all_problems:
            line = f"{p.name:>4}; "
            for name, solver in solvers:
                cf = CountedFunc(p.f)
                try:
                    result = solver(cf, p.a, p.b, p.value, eps)
                    line += f"{result:>{col_w}.15g}; "
                except Exception:
                    if args.no_error_supression:
                        raise
                    line += f"{'ERR':>{col_w}}; "
            print(line)
        print()

    if enable_fval:
        # Function values
        print("Function values")
        header = f"{'Func':>4}; " + "; ".join(f"{name:>{col_w}}" for name, _ in solvers)
        print(header)
        for p in all_problems:
            line = f"{p.name:>4}; "
            for name, solver in solvers:
                cf = CountedFunc(p.f)
                try:
                    result = solver(cf, p.a, p.b, p.value, eps)
                    true_answer = true_answers[p.name]
                    func_val = p.f(result)
                    if not true_answer.well_behaved:
                        line += f"{'?':>{col_w}}; "
                    else:
                        line += f"{func_val:>{col_w}.15g}; "
                except Exception:
                    if args.no_error_supression:
                        raise
                    line += f"{'ERR':>{col_w}}; "
            print(line)
        print()

    if enable_acc:
        # Root difference
        print("Root difference")
        header = f"{'Func':>4}; " + "; ".join(f"{name:>{col_w}}" for name, _ in solvers)
        print(header)
        for p in all_problems:
            line = f"{p.name:>4}; "
            for name, solver in solvers:
                cf = CountedFunc(p.f)
                try:
                    result = solver(cf, p.a, p.b, p.value, eps)
                    true_answer = true_answers[p.name]
                    accuracy = abs(true_answer.root - result)
                    if not true_answer.well_behaved:
                        line += f"{'?':>{col_w}}; "
                    else:
                        line += f"{accuracy:>{col_w}.15g}; "
                except Exception:
                    if args.no_error_supression:
                        raise
                    line += f"{'ERR':>{col_w}}; "
            print(line)
        print()

    if enable_nfev:
        # Function evaluation counts
        print("Function evaluations")
        header = f"{'Func':>4}; " + "; ".join(f"{name:>6}" for name, _ in solvers)
        print(header)
        total = [0] * len(solvers)
        for p in all_problems:
            line = f"{p.name:>4}; "
            for j, (name, solver) in enumerate(solvers):
                cf = CountedFunc(p.f)
                try:
                    solver(cf, p.a, p.b, p.value, eps)
                    total[j] += cf.count
                    line += f"{cf.count:>6}; "
                except Exception:
                    if args.no_error_supression:
                        raise
                    line += f"{'ERR':>6}; "
            print(line)

        # Print totals
        line = f"{'SUM':>4}; "
        for t in total:
            line += f"{t:>6}; "
        print(line)
        print()

def funcviz(args):
    func_name = args.func
    if func_name is None:
        raise Exception("--func is mandatory")
    func_x = args.func_x
    if func_x is None:
        func_x = true_answers[func_name].root
    for p in all_problems:
        if p.name == func_name:
            func = p.f
            break
    else:
        raise Exception(f"can't find func {func_name}")
    if func_x == 0:
        if args.func_size is None:
            start_x = -1e-10
            end_x = 1e-10
        else:
            start_x = -args.func_size
            end_x = args.func_size
    else:
        if args.func_size is None:
            size = func_x * 0.01
        else:
            size = args.func_size
        start_x = func_x - size / 2
        end_x = func_x + size / 2
    x = np.linspace(start_x, end_x, 1001)
    x = np.append(x, func_x)
    x.sort()
    y = np.array([func(float(xval)) for xval in x])
    plt.plot(x, y)
    plt.title(f"plot for {func_name}")
    plt.show()


def main(args):
    if args.mode == "bench":
        init_true_answers()
        bench(args)
    elif args.mode == "funcviz":
        init_true_answers()
        funcviz(args)
    elif args.mode == "termsearch":
        init_true_answers()
        termsearch(args)
    elif args.mode == "showsolutions":
        showsolutions(args)
    else:
        raise Exception("unknown --mode")



def parse_args():
    parser = argparse.ArgumentParser(
        prog="ModAB benchmark program"
    )

    parser.add_argument(
        "--mode",
        default="bench",
    )

    parser.add_argument(
        "--sections",
        type=str,
        default="roots,acc,nfev"
    )

    parser.add_argument(
        "-s",
        "--no-error-supression",
        action="store_true",
        help="don't suppress errors during bench",
    )

    parser.add_argument(
        "--func",
        help="for funcviz, what function to visualize?"
    )

    parser.add_argument(
        "--func-x",
        type=float,
        help="for funcviz, what x to visualize?"
    )

    parser.add_argument(
        "--func-size",
        type=float,
        help="for funcviz, what scale to visualize x on?"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
