import jax
import pytest
import triangles.env.mixed_action
import gymnasium as gym

from triangles.common import train_step, PolicyFactoryType, ExpConfig, SACStateFactory, Batch
from triangles.model import continuous, mixed_action
import jax.numpy as jnp


@pytest.mark.parametrize(["env_name", "state_factory"],
                         [("Pendulum-v1", continuous.policy_factory, continuous.sac_state_factory),
                          ("MixedAction2D-v0", mixed_action.policy_factory, mixed_action.sac_state_factory)])
def test_that_it_runs(env_name: str, policy_factory: PolicyFactoryType, state_factory: SACStateFactory):
    env = gym.make(env_name)
    policy = policy_factory(env)
    config = ExpConfig()
    sac_state = state_factory(config=config, env=env, policy=policy, rng_key=jnp.random.PRNGKey(0))

    # batch should be at least 2 elements
    [env.observation_space.sample(), env.observation_space.sample()]


    batch = Batch(obs=None, action=None, reward=None, terminated=None, next_obs=None)

    train_step(action_space=env.action_space,
               batch=batch,
               model_state=sac_state,
               config=config,
               rng_key=jnp.random.PRNGKey(0),
    )