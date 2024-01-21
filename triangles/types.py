# Copyright Craig Sherstan 2024
from typing import (
    Any,
    Optional,
    Union,
    Callable,
    Protocol,
    Mapping,
    runtime_checkable,
    cast,
)

import flax.struct
import numpy as np
from flax import linen as nn
from flax.core.scope import VariableDict, RNGSequences, CollectionFilter
from jax import Array

MappingArrayType = Mapping[str, Array]
AlphaType = MappingArrayType
ParamsType = Mapping[str, Any]
MetricsType = Mapping[str, Any]
NestedArray = Array | Mapping[str, "NestedArray"]
NestedNPArray = np.ndarray | Mapping[str, "NestedNPArray"]


class PolicyReturnType(flax.struct.PyTreeNode):
    """
    It's not fun creating containers for all the returns, but I do like the potential
    to reduce errors and make it easier to use with an IDE.

    I'll likely use this pattern throughout.
    """

    sampled_actions: NestedArray  # these are actions that are stochastic, for exploration
    log_probabilities: NestedArray  # log probabilities of the sampled_actions.
    deterministic_actions: NestedArray  # deterministic actions


class PolicyType(Protocol):
    """
    I've been asked why I would want PolicyType, it seems to add an unnecessary level of complication.

    Agreed. As it is right now, particularly with such a small codebase and only a single policy/model type this
    seems like overkill.

    The main idea here is that in the future I would have different kinds of models/policies and I want to have a
    protocol that defines how we interact with a policy in a consistent way.

    The nicest/most useful way to do this is something I'm still experimenting with. I'm not committed to this pattern.
    """
    def __call__(self, variables: VariableDict, observations: NestedArray, rng_key: Array) -> PolicyReturnType:
        raise NotImplementedError

class MetricWriter(Protocol):
    """
    Defines an interface for writing metrics
    """

    def scalar(self, tag: str, value: float, step: int) -> None:
        pass
