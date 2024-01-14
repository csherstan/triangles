import argparse
from functools import partial

from pathlib import Path

import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.common import ExpConfig, main


def env_factory(name: str, show: bool = False) -> gym.Env:
    return gym.make(name, render_mode="human" if show else "rgb_array")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
    args = parser.parse_args()

    config = ExpConfig(eval_frequency=500, num_eval_iterations=1)

    main(
        "mountain_car",
        config,
        args,
        partial(env_factory, name="MountainCarContinuous-v0"),
        policy_factory=policy_factory,
        sac_state_factory=sac_state_factory,
    )
