"""
Trains an agent on MixedAction2D. However, for this version the terminate action has been removed and
all of the discrete actions are converted to continuous actions.
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.sac import ExpConfig, main, get_main_parser
from triangles.env.mixed_action import ContinuousActionContinuingEnvWrapper


def env_factory(show: bool = False) -> gym.Env:
    env = gym.make(
        "MixedAction2D-v0",
        render_mode="human" if show else "rgb_array",
        continuous=True,
    )
    return ContinuousActionContinuingEnvWrapper(env)


if __name__ == "__main__":
    args = get_main_parser().parse_args()

    config = ExpConfig(eval_interval=500, num_eval_iterations=1)

    main(
        name="mixed_action_continuous_wrapper",
        config=config,
        args=args,
        env_factory=env_factory,
        policy_factory=policy_factory,
        sac_state_factory=sac_state_factory,
    )
