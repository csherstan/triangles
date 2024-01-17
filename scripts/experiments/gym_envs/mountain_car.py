# Copyright Craig Sherstan 2024
"""
Experiment for the continuous (continuous action space) version of MountainCar
"""
import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.sac import ExpConfig, main, get_main_parser


def env_factory(show: bool = False) -> gym.Env:
    return gym.make(
        "MountainCarContinuous-v0", render_mode="human" if show else "rgb_array"
    )


if __name__ == "__main__":
    args = get_main_parser().parse_args()

    config = ExpConfig(eval_interval=500, num_eval_iterations=1)

    main(
        name="mountain_car",
        config=config,
        args=args,
        env_factory=env_factory,
        policy_factory=policy_factory,
        sac_state_factory=sac_state_factory,
    )
