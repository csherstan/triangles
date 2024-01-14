import os

from triangles.model.mixed_action import sac_state_factory, policy_factory

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import argparse
from pathlib import Path

import gymnasium as gym
import structlog

from triangles.common import ExpConfig, main
import triangles.env.mixed_action

LOG = structlog.getLogger()


def env_factory(show: bool = False) -> gym.Env:
    return gym.make("MixedAction2D-v0", render_mode='human' if show else 'rgb_array')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
    args = parser.parse_args()

    config = ExpConfig(eval_frequency=500, num_eval_iterations=3, init_alpha={"mode": 0.5, "value": 0.5})
    main("discrete_action_terminating_env", config=config, args=args, env_factory=env_factory,
         policy_factory=policy_factory, sac_state_factory=sac_state_factory)
