import argparse
from functools import partial
from pathlib import Path

import gymnasium as gym

from triangles.classic_control import policy_factory, sac_state_factory
from triangles.common import ExpConfig, main


def env_factory(name: str, show: bool = False) -> gym.Env:
  return gym.make(name, render_mode='human' if show else 'rgb_array')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["train", "watch"], default="train")
  parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
  args = parser.parse_args()

  config = ExpConfig(max_replay_size=int(1e5))

  main("pendulum", config, args, partial(env_factory, name='Pendulum-v1'), policy_factory=policy_factory,
       sac_state_factory=sac_state_factory)
