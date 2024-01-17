# Copyright Craig Sherstan 2024
from typing import Optional, Iterator, Any, Dict

import jax
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from jax import Array, numpy as jnp


def rng_seq(
    *, seed: Optional[int] = None, rng_key: Optional[Array] = None
) -> Iterator[Array]:
    """
    Create a generator for using the jax rng keys. Not my idea. I saw it elsewhere, but so far I've liked the pattern.
    :param seed: Random Seed
    :param rng_key: Existing key to split
    :return:
    """
    assert seed is not None or rng_key is not None

    if rng_key is None:
        assert seed is not None
        rng_key = jax.random.PRNGKey(seed)

    while True:
        assert rng_key is not None
        rng_key, sub_key = jax.random.split(rng_key)
        yield sub_key


def as_float32(data: Any) -> np.ndarray:
    """
    Convert data to float32
    :param data: data to convert
    :return: A float32 numpy array
    """
    return np.asarray(data, dtype=np.float32)


def atleast_2d(data: Array) -> Array:
    """
    If smaller than 2d adds 1d to end. So (256,) -> (256,1)
    :param data: Data to convert
    :return: at least 2D data
    """

    if len(data.shape) < 2:
        data = jnp.expand_dims(data, -1)

    return data


def space_to_reverb_spec(
    space: spaces.Space,
) -> tf.TensorSpec | Dict[str, tf.TensorSpec]:

    """
    Converts a gym.space to a tf.TensorSpec for reverb buffer
    :param space: Space to convert
    :return: reverb spec of the space
    """

    if isinstance(space, spaces.Box):
        return tf.TensorSpec(shape=space.shape, dtype=tf.float32)
    elif isinstance(space, spaces.Discrete):
        # I don't like working with dimensionless vectors because they can lead to hidden broadcast errors, but
        # I want to keep the space comparable to the original environment
        return tf.TensorSpec(shape=(), dtype=tf.int64)
    elif isinstance(space, spaces.Dict):
        return {k: space_to_reverb_spec(v) for k, v in space.items()}

    raise Exception(f"Unsupported Type {type(space)}")
