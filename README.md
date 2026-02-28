# ModAB Root Finder

This package uses the ModAB algorithm to find roots in scalar problems.

It contains three solvers:

* modab_author: A Python implementation written by one of the paper's authors.
* modab_paper: A Cython translation of the original paper.
* modab_modern: A Cython translation of an updated version of the C# code.

The API is roughly the [root_scalar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html) API.

## Installation

```
pip install git+https://github.com/nickodell/modab_root_finder
```

## Example

```
>>> from modab_root_finder import root_scalar
>>> f = lambda x: x ** 3 - 0.5
>>> root_scalar(f, bracket=[-1, 1], method='modab_modern')
0.7937005259840997
```
