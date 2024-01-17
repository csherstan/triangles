# Copyright Craig Sherstan 2024
"""
Trains an agent on the Pendulum environment, which has a continuous action space
"""
import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.sac import ExpConfig, main, get_main_parser


def env_factory(show: bool = False) -> gym.Env:
    return gym.make("Pendulum-v1", render_mode="human" if show else "rgb_array")


if __name__ == "__main__":
    args = get_main_parser().parse_args()

    config = ExpConfig(max_replay_size=int(1e5))

    main(
        name="pendulum",
        config=config,
        args=args,
        env_factory=env_factory,
        policy_factory=policy_factory,
        sac_state_factory=sac_state_factory,
    )
