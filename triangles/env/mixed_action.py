# Copyright Craig Sherstan 2024

from typing import Any, SupportsFloat, Optional, Tuple, Dict, Union

import gymnasium as gym
import pygame
from gymnasium import Env, spaces
from gymnasium.core import ActionWrapper
import numpy as np
from pygame import Surface
from pygame.time import Clock

ObsType = np.ndarray
ActType = Dict[str, Union[np.ndarray, int, float]]


class MixedAction2D(Env[ObsType, ActType]):
    """
    Environment mixes discrete and continuous actions.

    The environment is a 2D x-y space where each axis goes from 0 to 1. The goal is to move the agent's cursor to
    the goal region.

    On each reset a new target position is sampled. The agent's goal is to move the cursor to the target position
    but it can only control one axis, either x or y, at a time. The agent can also signal that it is done.

    The environment can be set to `continuing`, meaning that it does not terminate, but will still timeout.

    target: R^2 \in [0, 1]
    action: Dict
      value: movement step [-1, 1] Continuous. In the environment this is mapped to [-0.1, 0.1]
      mode: discrete, 0: move on x-axis, 1: move on y-axis, 2: terminate

    observation:


    Reward: components
      distance (not terminating): -L2 norm of current position from target
      on termination:
        - if in the goal (green circle of radius 0.05) get +100
        - if not in the goal region -(100 +L2 norm)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, continuing: bool = False):
        self.continuing = continuing

        # if continuing the environment does not terminate except by timeout. The terminate action is not used.
        if self.continuing:
            self.action_space = spaces.Dict(
                {
                    "value": spaces.Box(low=-1, high=1, shape=(1,)),
                    "mode": spaces.Discrete(2),
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    "value": spaces.Box(low=-1, high=1, shape=(1,)),
                    "mode": spaces.Discrete(3),
                }
            )

        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,))

        self.current_pos = np.zeros(shape=(2,))
        self.target_pos = np.zeros(shape=(2,))

        # being within this radius is the goal
        self.target_radius = 0.05

        # action maxes are rescaled to this value.
        self.max_action = 0.1

        # used for rendering
        self.window_size = 500
        self.window: Optional[Surface] = None
        self.clock: Optional[Clock] = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        """
        Randomly draws a new starting position and target position
        :param seed: Random seed to use.
        :param options: Not used
        :return: Obs, empty dict
        """

        super().reset(seed=seed, options=options)

        self.target_pos = self.np_random.random(size=(2,))
        self.current_pos = self.np_random.random(size=(2,))

        return self._get_obs(), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        action_value = action["value"]  # the movement value
        terminated = False
        truncated = False

        # discrete action selection: either x or y axis or terminate
        match action["mode"]:
            # TODO: use enum
            case 0:
                # move on x-axis
                self.current_pos[1] += action_value * self.max_action
            case 1:
                # move on y-axis
                self.current_pos[0] += action_value * self.max_action
            case 2:
                # terminate
                terminated = True

                if self.continuing:
                    raise Exception(
                        "Environment set to continuing, but received terminated signal."
                    )

        self.current_pos = np.clip(self.current_pos, a_min=0.0, a_max=1.0)

        dist = np.linalg.norm(self.target_pos - self.current_pos)
        reward = -float(dist)
        if terminated:
            if dist < self.target_radius:
                reward = 100.0
            else:
                reward -= 100.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | list[np.ndarray] | None:  # type: ignore
        def x_y_to_pix(point: np.ndarray) -> Tuple[int, int]:
            return (int(self.window_size * point[0]), int(self.window_size * point[1]))

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )

            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pygame.draw.circle(
            canvas,
            color=(0, 255, 0),
            center=x_y_to_pix(self.target_pos),
            radius=self.window_size * self.target_radius,
        )

        pygame.draw.circle(
            canvas, color=(0, 0, 255), center=x_y_to_pix(self.current_pos), radius=4
        )

        if self.render_mode == "human":
            assert self.window
            assert self.clock
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self) -> np.ndarray:
        """
        :return: First two elements are the positional error and the final element is the distance from the target
        """
        return np.concatenate(
            [
                np.array(self.target_pos - self.current_pos, dtype=np.float32),
                np.array([np.linalg.norm(self.target_pos - self.current_pos)]),
            ]
        )


gym.register(
    "MixedAction2D-v0",
    entry_point="triangles.env.mixed_action:MixedAction2D",
    nondeterministic=False,
    max_episode_steps=100,
)


# The generics don't seem right here. Looking at the ActionWrapper it wants 4, but mypy says it needs only 3.
class ContinuousActionContinuingEnvWrapper(ActionWrapper[ObsType, np.ndarray, ActType]):
    def __init__(self, env: gym.Env):
        """
        In this setting we use a continuing environment - it does not terminate, only times out.
        So only modes 1 and 2 are used: move x, move y.
        All actions are mapped to a continuous action space

        :param env:
        """
        super().__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def action(self, action: np.ndarray) -> ActType:
        mode = int(action[0] >= 0)

        return {"mode": mode, "value": action[1]}


class ContinuousActionTerminatingEnvWrapper(
    ActionWrapper[ObsType, np.ndarray, ActType]
):
    def __init__(self, env: gym.Env):
        """
        In this setting we use a terminating environment - the agent can signal that it is finished.
        All actions are mapped to a continuous action space:
        action[0] - If greater > 0.0 terminate
        action[1] - If greater > 0.0 move y axis, else move x axis
        action[2] - Movement value

        :param env:
        """
        super().__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def action(self, action: np.ndarray) -> ActType:
        axis_movement = int(action[1] >= 0)
        terminate = int(action[0] >= 0)

        mode = 2 if terminate else axis_movement

        return {"mode": mode, "value": action[2]}


if __name__ == "__main__":
    env = gym.make("MixedAction2D-v0", render_mode="human")

    while True:
        obs, _ = env.reset()
        the_return = 0.0
        while True:
            action = env.action_space.sample()
            action["mode"] = 0
            obs, reward, terminated, truncated, info = env.step(action)
            the_return += float(reward)

            if truncated or terminated:
                print(f"{the_return=}")
                break
