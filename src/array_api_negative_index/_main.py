from typing import Any, cast

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace


def to_symmetric(
    input: Array,  # noqa: A002
    /,
    *,
    axis: int = 0,
    asymmetric: bool = False,
    conjugate: bool = False,
) -> Array:
    """
    Extend a tensor to its opposite symmetric form.

    Parameters
    ----------
    input : Array
        The input tensor.
    axis : int, optional
        The axis to extend, by default 0
    asymmetric : bool, optional
        If True, the input tensor is multiplied by -1, by default False
    conjugate : bool, optional
        If True, the input tensor is conjugated, by default False

    Returns
    -------
    Array
        The symmetric tensor.
        Forall a < input.shape[axis], result[-a] = result[a].

    """
    xp = array_namespace(cast("Any", input))
    input_to_symmetric = input
    if asymmetric:
        input_to_symmetric = -input_to_symmetric
    if conjugate:
        input_to_symmetric = xp.conj(input_to_symmetric)
    axis %= input.ndim
    input_to_symmetric = input_to_symmetric[(slice(None),) * axis + (slice(1, None), ...)]
    input_to_symmetric = xp.flip(input_to_symmetric, axis=axis)
    return xp.concat([input, input_to_symmetric], axis=axis)


def arange_asymmetric(
    stop: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    dtype: Any = None,
    device: Any = None,
) -> Array:
    """
    Create a symmetric array with values from -stop+1 to stop-1.

    Parameters
    ----------
    stop : int
        The stop value (exclusive).
    dtype : Any, optional
        The desired data type of the output array, by default None
    device : Any, optional
        The device on which to create the array, by default None
    xp : ArrayNamespaceFull, optional
        The array namespace to use, by default None

    Returns
    -------
    Array
        An array [0, 1, ..., stop-1, -stop+1, ..., -1].
        The shape is (2 * stop - 1,).

    """
    return cast(
        "Array",
        to_symmetric(
            xp.arange(stop, dtype=dtype, device=device),
            asymmetric=True,
        ),
    )


def flip_symmetric(input: Array, /, *, axis: int = 0) -> Array:  # noqa: A002
    """
    Flip a symmetric array.

    Parameters
    ----------
    input : Array
        The input array.
    axis : int, optional
        The axis to flip, by default 0

    Returns
    -------
    Array
        The flipped array.
        Forall a < input.shape[axis] result[-a-1] = result[a] = input[a] = input[-a-1]

    """
    xp = array_namespace(cast("Any", input))
    axis %= input.ndim
    zero = input[
        (
            slice(
                None,
            ),
        )
        * axis
        + (slice(0, 1), ...)
    ]
    nonzero = input[(slice(None),) * axis + (slice(1, None), ...)]
    return xp.concat([zero, xp.flip(nonzero, axis=axis)], axis=axis)
