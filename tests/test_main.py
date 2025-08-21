import pytest
from array_api._2024_12 import ArrayNamespaceFull

from array_api_negative_index import flip_symmetric, to_symmetric


def test_flip_symmetric(xp: ArrayNamespaceFull) -> None:
    array = xp.asarray([0, 1, 2, 3, 4])
    assert xp.all(flip_symmetric(array) == xp.asarray([0, 4, 3, 2, 1]))
    assert xp.all(flip_symmetric(flip_symmetric(array)) == array)


@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric_manual(
    asymmetric: bool, conjugate: bool, xp: ArrayNamespaceFull
) -> None:
    array = xp.asarray([0, 1 + 1j, 2 - 1j])
    result = to_symmetric(xp.asarray(array), asymmetric=asymmetric, conjugate=conjugate)
    if asymmetric and conjugate:
        expected = [0, 1 + 1j, 2 - 1j, -2 - 1j, -1 + 1j]
    elif asymmetric and not conjugate:
        expected = [0, 1 + 1j, 2 - 1j, -2 + 1j, -1 - 1j]
    elif not asymmetric and conjugate:
        expected = [0, 1 + 1j, 2 - 1j, 2 + 1j, 1 - 1j]
    else:
        expected = [0, 1 + 1j, 2 - 1j, 2 - 1j, 1 + 1j]
    assert xp.all(result == xp.asarray(expected))
