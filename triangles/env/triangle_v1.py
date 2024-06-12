from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from triangles.env.triangle import TriangleEnv, ActType, ObsType

from PIL import Image, ImageDraw


class AddOnlyTriangleEnvWrapper(TriangleEnv):

    """
    I find it interesting how much the existing code base is impacting how I try to solve this problem.
    I'm limiting my solutions to what allows me to reuse code as much as possible.
    This might end up being a bad decision.

    I certainly see how much existing infrastructure can become a hindrance to rapid development.
    """

    def __init__(self, width: int, height: int, max_triangles: int = 20):
        super().__init__(width=width, height=height)
        self.max_triangles = max_triangles  # really this has the same effect as max_episode_steps

        self.action_space = spaces.Box(low=0, high=1, shape=(10,))
        self.observation_space = spaces.Dict(
            {
                "error": spaces.Box(low=-1, high=1, shape=(self.width, self.height), dtype=np.float32),
                "triangles": spaces.Box(low=-1, high=1, shape=(self.max_triangles, 10), dtype=np.float32),
                "triangle_count": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:

        obs, info = super().reset(seed=seed, options=options)

        obs = self.observation(obs)

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.action(action)

        observation, reward, terminated, truncated, info = super().step(action)

        if len(self.triangles) == self.max_triangles:
            terminated = True

        observation = self.observation(observation)

        return observation, reward, terminated, truncated, info

    def action(self, action: WrapperActType) -> ActType:
        return {
            "data": {
                "index": 0,
                "triangle": action
            },
            "op": TriangleEnv.Op.ADD
        }

    def observation(self, observation: ObsType) -> WrapperObsType:
        triangles = np.zeros((self.max_triangles, 10))
        if len(self.triangles) > 0:
            triangles[0:len(self.triangles)] = np.row_stack([t.array for t in self.triangles])

        return {
            "error": observation["error"],
            "triangles": triangles,
            "triangle_count": len(self.triangles),
        }


gym.register("triangles-v1", entry_point="triangles.env.triangle_v1:AddOnlyTriangleEnvWrapper",
             nondeterministic=False,
             max_episode_steps=20,
             # max_triangles=20,
             )


if __name__ == "__main__":

    width = 300
    height = 600
    channels = 3
    env = AddOnlyTriangleEnvWrapper(width=width, height=height)
    image = np.zeros(shape=(height, width, channels))

    init_obs, _ = env.reset(options={"target": image})
    for i in range(20):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        img: np.ndarray = env.render()
        Image.fromarray(img).show()