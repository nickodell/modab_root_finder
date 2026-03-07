import pytest

from modab_root_finder import root_scalar
from numpy.testing import assert_allclose
from modab_root_finder.tests.util import debug_func


all_methods = [
    'modab_author',
    'modab_paper',
    'modab_modern',
    'modab_refactor',
]


@pytest.mark.parametrize("method", all_methods)
def test_cubic(method):
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method=method)
    assert_allclose(root, 0.5 ** (1/3))


@pytest.mark.parametrize("method", all_methods)
def test_f24(method):
    def f(x):
        return (x + 2) * (x + 1) * (x - 3)**3
    root = root_scalar(f, bracket=[2.6, 4.6], xtol=1e-14, method=method)
    # This function has 3 roots, but only one is inside range
    assert_allclose(root, 3)
