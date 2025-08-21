import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(scope="session", params=["numpy", "torch"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    backend = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp
    elif backend == "torch":
        from array_api_compat import torch as xp
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
