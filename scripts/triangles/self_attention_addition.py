import os

import jax
import optax
from jax import Array
import jax.numpy as jnp

from triangles.env.triangle_v1 import AddOnlyTriangleEnvWrapper
from triangles.model.triangles_nn import PolicyHead, Core, QHead

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import argparse

from pathlib import Path

import gymnasium as gym

from triangles.model.continuous import policy_factory, sac_state_factory
from triangles.common import ExpConfig, main, PolicyType, QTrainState, ModelState
from triangles.env.mixed_action import ContinuousActionContinuingEnvWrapper


def env_factory(show: bool = False) -> gym.Env:
    env = gym.make(
        "triangles-v0",
        render_mode="human" if show else "rgb_array",
        continuous=True,
    )
    return AddOnlyTriangleEnvWrapper(env)

class Model():

    def __init__(self, env):
        width = 300
        height = 600
        channels = 3
        image = jnp.zeros(shape=(height, width, channels))

        init_obs, _ = env.reset(options={"target": image})
        action = env.action_space.sample()

        init_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), init_obs)
        action = jax.tree_map(lambda x: jnp.expand_dims(x, 0), action)

        core = Core()
        core_output, core_variables = core.init_with_output(jax.random.PRNGKey(0), init_obs)
        q_function = QHead()

        q_output, q_variables = q_function.init_with_output(jax.random.PRNGKey(0), core_output, action)

        policy = PolicyHead()
        policy_output, policy_variables = policy.init_with_output(jax.random.PRNGKey(0), core_output,
                                                                  jax.random.PRNGKey(0))

        QTrainState.create(
            apply_fn=(),
            params=None,
            tx=optax.adam(
                learning_rate=config.q_learning_rate,
                b1=config.adam_beta_1,
                b2=config.adam_beta_2,
            ),
            target_params=q_variables["params"],
        )

    def policy_factory(self, env: gym.Env) -> PolicyHead:
        pass

    def sac_state_factory(self,
        config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array
    ) -> ModelState:
        pass

    def q_state(self):
        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
    args = parser.parse_args()

    config = ExpConfig(eval_frequency=500, num_eval_iterations=1)

    model = Model()

    main(
        "mixed_action_continuous_wrapper",
        config,
        args,
        env_factory,
        policy_factory=model.policy_factory,
        state_factory=model.sac_state_factory,
    )
