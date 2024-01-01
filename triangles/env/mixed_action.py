from typing import Any, SupportsFloat

from gymnasium import Env, spaces
from gymnasium.core import ObsType, ActType
import numpy as np


class MixedAction2D(Env):

  def __init__(self):
    self.action_space = spaces.Dict({"value": spaces.Box(low=-1, high=1, shape=(1,)), "mode": spaces.Discrete(3)})
    self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

  def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

    super().reset(seed=seed, options=options)

    self.target = np.random.random(size=(2,))
    self.current_pos = np.random.random(size=(2,))

    return self._get_obs(), {}


  def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

    action_value = action["value"]
    terminated = False
    truncated = False

    match action["mode"]:
      case 0:
        # move on x-axis
        self.current_pos[1] += action_value
      case 1:
        # move on y-axis
        self.current_pos[0] += action_value
      case 2:
        # terminate
        terminated = True

    self.current_pos = np.clip(self.current_pos, a_min=0.0, a_max=1.0)

    reward = -1 - np.linalg.norm(self.target - self.current_pos)

    return self._get_obs(), reward, terminated, truncated, {}

  def _get_obs(self):
    return self.target - self.current_pos
