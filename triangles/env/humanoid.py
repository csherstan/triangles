from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import WrapperObsType


class Humanoid(gym.Wrapper):
    def __init__(self):
        env = gym.make("Humanoid-v4")
        self.action_names = [
            "abdomen_y",
            "abdomen_z",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ]

        super().__init__(env)

    def step(self, action) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["joint_rewards"] = {name: a**2 for name, a in zip(self.action_names, action)}

        return obs, reward, terminated, truncated, info


gym.register(
     id="triangles/Humanoid-v4",
     entry_point="triangles.env.humanoid:Humanoid",
)
