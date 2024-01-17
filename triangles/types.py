# Copyright Craig Sherstan 2024
from typing import Any, Optional, Union, Callable, Protocol, Mapping

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
    sampled_actions: NestedArray    # these are actions that are stochastic, for exploration
    log_probabilities: NestedArray  # log probabilities of the sampled_actions.
    deterministic_actions: NestedArray  # deterministic actions


class PolicyType(nn.Module):
    def __call__(self, observations: NestedArray, rng_key: Array) -> PolicyReturnType:
        raise NotImplementedError

    def apply(
      self,
      variables: VariableDict,
      *args: Any,
      rngs: Optional[RNGSequences] = None,
      method: Union[Callable[..., Any], str, None] = None,
      mutable: CollectionFilter = False,
      capture_intermediates: Union[
          bool, Callable[['nn.Module', str], bool]
      ] = False,
      **kwargs: Any,
    ) -> PolicyReturnType:
        """
        I'm just explicitly defining this so I can set the return type
        """
        raise NotImplementedError





class MetricWriter(Protocol):
    """
    Defines an interface for writing metrics
    """

    def scalar(self, tag: str, value: float, step: int) -> None:
        pass
