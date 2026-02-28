from modab_root_finder import root_scalar
import numpy as np


def test_author():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_author')
    assert np.isclose(0.5 ** (1/3), root)


def test_paper():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_paper')
    assert np.isclose(0.5 ** (1/3), root)


def test_modern():
    f = lambda x: x ** 3 - 0.5
    root = root_scalar(f, bracket=[-1, 1], method='modab_modern')
    assert np.isclose(0.5 ** (1/3), root)
