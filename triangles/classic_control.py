"""
This is a training script to first train gym agents in the classic control setting to
validate my SACv2 implementation:

- Critic uses 2 Q functions
- Adaptive entropy loss weight

Also, this version is an asynchronous implementation where there are multiple rollout workers and a single trainer.
The rollout workers are deployed as python processes that read in the latest policy parameters, collect an episode
of experience, and then write the data to the reverb replay buffer. A single trainer process samples batches of data
from the reverb replay buffer, performs a training step, and sends the updated model to all rollout workers.
There is also an eval process that gets triggered at certain model clock intervals.
Note, that at present one training step is one model clock step.

The main reason for choosing this implementation is that it mirrors the approach that we used with GT Sophy and is
familiar. However, I did not have the experience of writing that code myself, so I wanted more first-hand experience
with SAC.

Alternatives I might considered for rollout collection:
- VectorizedEnvs
- EnvPool: https://envpool.readthedocs.io/


---
Import issues:
1. At the time of writing jaxlib 0.4.23 has a bug from the xla prject that prevents it from obeying
XLA_PYTHON_CLIENT_PREALLOCATE Because of this I have downgraded to 0.4.19

2. The orbax checkpointer has some bugs where the API is not matching. To correct this it would be nice to upgrade
orbax to the latest, but the latest requires ml-types greater than 0.3. However, the current release of tf requires
ml-types==0.2. The nightly build of tf allows 0.3, but going down that route broke all sorts of things. So instead
I've dropped orbax and am using flax's own checkpoint system.

3. Pycharm IDE doesn't play nice with Flax dataclasses: https://youtrack.jetbrains.com/issue/PY-54560

TODO: factor loss function into 3 functions.
TODO: log collection returns
TODO: abstract metric writing
TODO: abstract artifact collection
TODO: define return types as named tuples or dataclasses
TODO: mypy

One thing I've learned is that working with complex structures with jax, reverb, gym.spaces, etc., is a pain.
I would probably be better off to just flatten everything at the environment interface.


"""

import os

from triangles.common import rng_seq, ExpConfig, QTrainState, SACModelState

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from typing import Tuple

import distrax
import gymnasium as gym
import optax
import structlog
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array, numpy as jnp

LOG = structlog.getLogger()


class Policy(nn.Module):
  action_size: int

  @nn.compact
  def __call__(self, observations: Array, rng_key: Array) -> Tuple[Array, Array, Array]:
    observations = jnp.atleast_2d(observations)  # add batch dim if not present
    rng_gen = rng_seq(rng_key=rng_key)

    # TODO: for the moment I'm hardcoding some vals just for pendulum
    x = nn.Sequential([
      nn.Dense(256),
      nn.relu,
      nn.Dense(256),
      nn.relu,
    ])(observations)

    means = nn.Dense(self.action_size)(x)
    # log_std_dev is defined on [-inf, inf]
    log_std_dev = nn.Dense(self.action_size)(x)
    std_dev = jnp.exp(log_std_dev)

    norm = distrax.MultivariateNormalDiag(loc=means, scale_diag=std_dev)
    dist = distrax.Transformed(distribution=norm, bijector=distrax.Block(distrax.Tanh(), ndims=1))

    actions, action_log_prob = dist.sample_and_log_prob(seed=next(rng_gen))

    return actions, jnp.expand_dims(action_log_prob, -1), jnp.tanh(means)


class QFunction(nn.Module):

  @nn.compact
  def __call__(self, states: Array, actions: Array) -> Array:
    inputs = jnp.concatenate([states, actions], axis=1)

    return nn.Sequential([
      nn.Dense(256),
      nn.relu,
      nn.Dense(256),
      nn.relu,
      nn.Dense(1)
    ])(inputs)


def create_policy_state(env: gym.Env, policy, config: ExpConfig, rng_key: Array) -> TrainState:
  rng_gen = rng_seq(rng_key=rng_key)
  init_samples = [env.observation_space.sample(), env.observation_space.sample()]
  output, policy_variables = policy.init_with_output(next(rng_gen), jnp.array(init_samples), next(rng_gen))
  policy_state = TrainState.create(
    apply_fn=policy.apply,
    params=policy_variables["params"],
    tx=optax.adam(learning_rate=config.policy_learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2)
  )

  return policy_state


def create_q_state(env: gym.Env, config: ExpConfig, rng_key: Array) -> QTrainState:
  rng_gen = rng_seq(rng_key=rng_key)
  q = QFunction()
  init_samples = jnp.array([env.observation_space.sample(), env.observation_space.sample()])
  init_actions = jnp.array([env.action_space.sample(), env.action_space.sample()])
  output, q_variables = q.init_with_output(next(rng_gen), states=jnp.array(init_samples), actions=init_actions)
  q_state = QTrainState.create(
    apply_fn=q.apply,
    params=q_variables["params"],
    tx=optax.adam(learning_rate=config.q_learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2),
    target_params=q_variables["params"]
  )

  return q_state


def policy_factory(env: gym.Env) -> Policy:
  action_size = env.action_space.shape[0]
  # TODO: hardcoding this is bad
  return Policy(action_size=action_size)


def sac_state_factory(config: ExpConfig, env: gym.Env, policy, rng_key: Array) -> SACModelState:
  rng_gen = rng_seq(rng_key=rng_key)
  policy_state = create_policy_state(env=env, policy=policy, config=config, rng_key=next(rng_gen))
  q1_state = create_q_state(env, config, next(rng_gen))
  q2_state = create_q_state(env, config, next(rng_gen))

  # although alpha is a scalar, it needs to have a dimension for jax.grad to be happy
  alpha_params = {"alpha": jnp.array([config.alpha])}

  return SACModelState(policy_state=policy_state,
                       q1_state=q1_state,
                       q2_state=q2_state,
                       alpha_params=alpha_params,
                       alpha_optimizer_params=optax.adam(learning_rate=config.alpha_lr).init(alpha_params),
                       model_clock=jnp.array(0, dtype=jnp.int32), )


