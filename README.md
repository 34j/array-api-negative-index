# Array API Negative Index

<p align="center">
  <a href="https://github.com/34j/array-api-negative-index/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/array-api-negative-index/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://array-api-negative-index.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/array-api-negative-index.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/array-api-negative-index">
    <img src="https://img.shields.io/codecov/c/github/34j/array-api-negative-index.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/array-api-negative-index/">
    <img src="https://img.shields.io/pypi/v/array-api-negative-index.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/array-api-negative-index.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/array-api-negative-index.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://array-api-negative-index.readthedocs.io" target="_blank">https://array-api-negative-index.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/array-api-negative-index" target="_blank">https://github.com/34j/array-api-negative-index </a>

---

Utils for indexing arrays with {-n, -(n-1), ..., -1, 0, 1, ..., n-1, n} using array API.

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install array-api-negative-index
```

## Usage

This package provides a utility for packing array elements in the order [0, 1, 2, ..., n, -n-1, ..., -1].
This allows one to access the `i`-th element by `array[i]`, regardless of whether `i` is positive, 0 or negative, thanks to "Negative Indexing" in array API.
This is useful in practical applications such as expressing the order `m` of spherical harmonics.

```python
from array_api_negative_index import to_symmetric, flip_symmetric, arange_asymmetric
```

### `arange_asymmetric()`

```python
import numpy as np

a = arange_asymmetric(3, xp=np)
print(a)
```

```text
[ 0  1  2 -2 -1]
```

Not to confuse with `np.arange(-stop + 1, stop)`!
This caused me to create bugs many times.

```python
a = np.arange(-2, 3)
print(a)
```

```text
[-2, -1, 0, 1, 2]
```

### `flip_symmetric()`

```python
b = flip_symmetric(a)
print(f"{a} -> (flip) -> {b}")
```

```text
[ 0  1  2 -2 -1] -> (flip) -> [ 0 -1 -2  2  1]
```

### `to_symmetric()`

```python
c = np.asarray([0, 3, 5])
d = to_symmetric(c, asymmetric=True, conjugate=False)
print(f"{c} -> (to_symmetric) -> {d}")
```

```text
[0 3 5] -> (to_symmetric) -> [ 0  3  5 -5 -3]
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
