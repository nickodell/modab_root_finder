from modab_root_finder import root_scalar
from numpy.testing import assert_allclose



def test_author():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_author')
    assert_allclose(root, 0.5 ** (1/3))


def test_paper():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_paper')
    assert_allclose(root, 0.5 ** (1/3))


def test_modern():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_modern')
    assert_allclose(root, 0.5 ** (1/3))
