import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import argparse

from pathlib import Path

import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.common import ExpConfig, main
from triangles.env.mixed_action import ContinuousActionTerminatingEnvWrapper


def env_factory(show: bool = False) -> gym.Env:
    env = gym.make("MixedAction2D-v0", render_mode="human" if show else "rgb_array")
    return ContinuousActionTerminatingEnvWrapper(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
    args = parser.parse_args()

    config = ExpConfig(eval_frequency=500, num_eval_iterations=3)

    main(
        "continuous_action_terminating_env",
        config,
        args,
        env_factory,
        policy_factory=policy_factory,
        sac_state_factory=sac_state_factory,
    )
