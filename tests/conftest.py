from typing import Any, cast

import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(scope="session", params=["numpy", "torch"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    backend = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp  # noqa: PLC0415
    elif backend == "torch":
        from array_api_compat import torch as xp  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return cast("Any", xp)
