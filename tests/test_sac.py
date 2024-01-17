# Copyright Craig Sherstan 2024
import pytest

from triangles.sac import compute_entropy_bonus
import jax.numpy as jnp

from triangles.types import AlphaType, MappingArrayType


@pytest.mark.parametrize(["alpha_params", "logits", "expected"], [
    [{"alpha": jnp.log(jnp.array([0.5]))}, jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[0.5], [1.0], [1.5]])],
    [{"alpha": {"0.5": jnp.log(jnp.array([0.5])), "0.1": jnp.log(jnp.array([0.1]))}},
     {"0.5": jnp.array([[1.0], [2.0], [3.0]]), "0.1": jnp.array([[1.0], [2.0], [3.0]])},
     jnp.array([[0.5+0.1], [1.0+0.2], [1.5+0.3]])
     ],
])
def test_compute_entropy_bonus(alpha_params: AlphaType, logits: MappingArrayType, expected: MappingArrayType):
    received = compute_entropy_bonus(alpha_params, logits)
    assert jnp.allclose(expected, received)

