import argparse
from functools import partial

from pathlib import Path
from typing import Tuple

import gymnasium as gym

from triangles.classic_control import ExpConfig, main


def env_factory(name: str, show: bool = False) -> gym.Env:
  return gym.make(name, render_mode='human' if show else 'rgb_array')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["train", "watch"], default="train")
  parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
  parser.add_argument("--env_name", type=str, default="Pendulum-v1")
  args = parser.parse_args()

  config = ExpConfig()

  main(config, args, partial(env_factory, name=args.env_name))
