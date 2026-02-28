from ._core import echo
from .modab_from_proektsoftbg_encoded import modab_from_proektsoftbg
from .modab_from_paper import modab_from_paper
from .modab_modern_impl import modab_modern_impl
import functools

__all__ = ["root_scalar"]
__version__ = "0.0.0dev0"


methods = {
    'modab_author': modab_from_proektsoftbg,
    'modab_paper': modab_from_paper,
    'modab_modern': modab_modern_impl,
}


def root_scalar(
    f, args=(), method=None, bracket=None, fprime=None,
    fprime2=None, x0=None, x1=None, xtol=1e-12, rtol=0,
    maxiter=100, options=None,
):
    if method is None:
        method = "modern"
    if bracket is None:
        raise ValueError("A bracket is required for this method.")
    if fprime is not None:
        raise ValueError("fprime will be ignored by this method.")
    if fprime2 is not None:
        raise ValueError("fprime will be ignored by this method.")
    if x0 is not None:
        raise ValueError("Specify the bracket via `bracket`, not x.")
    if x1 is not None:
        raise ValueError("Specify the bracket via `bracket`, not x.")
    if rtol != 0:
        raise NotImplementedException("rtol not currently implemented.")
    if not isinstance(args, tuple):
        args = (args,)
    if args != ():
        f = functools.partial(f, *args)
    if len(bracket) != 2:
        raise ValueError("bracket must have 2 elements")
    if method not in methods:
        raise ValueError(f"method {method} not found. Options are {method.keys()}")

    method_func = methods[method]
    lo, hi = bracket
    root = method_func(f, lo, hi, xtol, maxiter)
    return root
