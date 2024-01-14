from typing import Tuple, cast

import distrax
import gymnasium as gym
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array, numpy as jnp
from gymnasium import spaces

from triangles.common import rng_seq, ExpConfig, QTrainState, SACModelState, PolicyType


class Policy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(
        self, observations: Array, rng_key: Array
    ) -> Tuple[Array, Array, Array]:
        observations = jnp.atleast_2d(observations)  # add batch dim if not present
        rng_gen = rng_seq(rng_key=rng_key)

        # TODO: for the moment I'm hardcoding some vals just for pendulum
        x = nn.Sequential(
            [
                nn.Dense(256),
                nn.relu,
                nn.Dense(256),
                nn.relu,
            ]
        )(observations)

        means = nn.Dense(self.action_size)(x)
        # log_std_dev is defined on [-inf, inf]
        log_std_dev = nn.Dense(self.action_size)(x)
        std_dev = jnp.exp(log_std_dev)

        norm = distrax.MultivariateNormalDiag(loc=means, scale_diag=std_dev)
        dist = distrax.Transformed(
            distribution=norm, bijector=distrax.Block(distrax.Tanh(), ndims=1)
        )

        actions, action_log_prob = dist.sample_and_log_prob(seed=next(rng_gen))

        return actions, jnp.expand_dims(action_log_prob, -1), jnp.tanh(means)


class QFunction(nn.Module):
    @nn.compact
    def __call__(self, states: Array, actions: Array) -> Array:
        inputs = jnp.concatenate([states, actions], axis=1)

        return cast(
            Array,
            nn.Sequential(
                [nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(1)]
            )(inputs),
        )


def create_policy_state(
    env: gym.Env, policy: PolicyType, config: ExpConfig, rng_key: Array
) -> TrainState:
    rng_gen = rng_seq(rng_key=rng_key)
    init_samples = [env.observation_space.sample(), env.observation_space.sample()]
    output, policy_variables = policy.init_with_output(
        next(rng_gen), jnp.array(init_samples), next(rng_gen)
    )
    policy_state: TrainState = TrainState.create(
        apply_fn=policy.apply,
        params=policy_variables["params"],
        tx=optax.adam(
            learning_rate=config.policy_learning_rate,
            b1=config.adam_beta_1,
            b2=config.adam_beta_2,
        ),
    )

    return policy_state


def create_q_state(env: gym.Env, config: ExpConfig, rng_key: Array) -> QTrainState:
    rng_gen = rng_seq(rng_key=rng_key)
    q = QFunction()
    init_samples = jnp.array(
        [env.observation_space.sample(), env.observation_space.sample()]
    )
    init_actions = jnp.array([env.action_space.sample(), env.action_space.sample()])
    output, q_variables = q.init_with_output(
        next(rng_gen), states=jnp.array(init_samples), actions=init_actions
    )
    q_state: QTrainState = QTrainState.create(
        apply_fn=q.apply,
        params=q_variables["params"],
        tx=optax.adam(
            learning_rate=config.q_learning_rate,
            b1=config.adam_beta_1,
            b2=config.adam_beta_2,
        ),
        target_params=q_variables["params"],
    )

    return q_state


def policy_factory(env: gym.Env) -> Policy:
    assert isinstance(env.action_space, spaces.Box)
    action_size = env.action_space.shape[0]
    return Policy(action_size=action_size)


def sac_state_factory(
    config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array
) -> SACModelState:
    rng_gen = rng_seq(rng_key=rng_key)
    policy_state = create_policy_state(
        env=env, policy=policy, config=config, rng_key=next(rng_gen)
    )
    q1_state = create_q_state(env, config, next(rng_gen))
    q2_state = create_q_state(env, config, next(rng_gen))

    # although alpha is a scalar, it needs to have a dimension for jax.grad to be happy
    alpha_params = {"alpha": jnp.array([config.init_alpha])}

    return SACModelState(
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        alpha_params=alpha_params,
        alpha_optimizer_params=optax.adam(learning_rate=config.alpha_lr).init(
            alpha_params
        ),
        model_clock=jnp.array(0, dtype=jnp.int32),
    )
