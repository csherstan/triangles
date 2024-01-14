from typing import Tuple

import distrax
import flax.linen
import gymnasium as gym
import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax import Array, numpy as jnp

from triangles.common import DictArrayType, rng_seq, PolicyType, ExpConfig, QTrainState, stack_dict_jnp, \
    convert_space_to_jnp, SACModelState


class Policy(nn.Module):
    action_space: spaces.Space

    @nn.compact
    def __call__(self, observations: Array, rng_key: Array) -> Tuple[DictArrayType, DictArrayType, DictArrayType]:
        discrete_modes = self.action_space["mode"].n
        observations = jnp.atleast_2d(observations)
        rng_gen = rng_seq(rng_key=rng_key)

        intermediate_output = nn.Sequential([
            nn.Dense(16),
            nn.relu,
            nn.Dense(16),
            nn.relu,
        ])(observations)

        # TODO: I'm making an assumption that the distrax Softmax can have gradients taken through it.
        discrete_logits = nn.Sequential([nn.Dense(discrete_modes)])(intermediate_output)
        discrete_dist = distrax.Softmax(discrete_logits)

        # outputs are indexes, not one-hot vectors.
        discrete_samples, discrete_log_prob = discrete_dist.sample_and_log_prob(seed=next(rng_gen))
        discrete_exploit = jnp.argmax(discrete_logits, axis=1)

        # continuous
        one_hot_mode = jax.lax.stop_gradient(
            flax.linen.activation.one_hot(discrete_samples, num_classes=discrete_modes))
        continuous_action_input = jnp.concatenate([intermediate_output, one_hot_mode], axis=1)
        continuous_action_size = 1

        continuous_action_neck = nn.Sequential([nn.Dense(16, nn.relu), nn.Dense(16, nn.relu)])(continuous_action_input)

        means = nn.Dense(continuous_action_size)(continuous_action_neck)
        log_std_dev = nn.Dense(continuous_action_size)(continuous_action_neck)
        std_dev = jnp.exp(log_std_dev)

        norm = distrax.MultivariateNormalDiag(loc=means, scale_diag=std_dev)
        dist = distrax.Transformed(distribution=norm, bijector=distrax.Block(distrax.Tanh(), ndims=1))
        continuous_samples, continuous_log_prob = dist.sample_and_log_prob(seed=next(rng_gen))

        return ({"mode": discrete_samples, "value": continuous_samples},
                {"mode": discrete_log_prob, "value": continuous_log_prob},
                {"mode": discrete_exploit, "value": means})


def create_policy_state(env: gym.Env, policy: PolicyType, config: ExpConfig, rng_key: Array) -> TrainState:
    rng_gen = rng_seq(rng_key=rng_key)
    init_samples = [env.observation_space.sample(), env.observation_space.sample()]
    output, policy_variables = policy.init_with_output(next(rng_gen), jnp.array(init_samples), next(rng_gen))
    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy_variables["params"],
        tx=optax.adam(learning_rate=config.policy_learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2)
    )

    return policy_state


class QFunction(nn.Module):
    action_space: spaces.Space

    @nn.compact
    def __call__(self, observations: Array, actions: DictArrayType) -> Array:
        mode = jnp.squeeze(nn.activation.one_hot(actions["mode"], num_classes=self.action_space["mode"].n))
        inputs = jnp.concatenate([mode, actions["value"]], axis=1)

        return nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(64),
            nn.relu,
            nn.Dense(1)
        ])(inputs)


def create_q_state(env: gym.Env, config: ExpConfig, rng_key: Array) -> QTrainState:
    rng_gen = rng_seq(rng_key=rng_key)
    q = q_factory(env)
    init_samples = jnp.array([env.observation_space.sample(), env.observation_space.sample()])
    init_actions = stack_dict_jnp(
        [convert_space_to_jnp(env.action_space.sample()), convert_space_to_jnp(env.action_space.sample())])
    output, q_variables = q.init_with_output(next(rng_gen), observations=jnp.array(init_samples), actions=init_actions)
    q_state = QTrainState.create(
        apply_fn=q.apply,
        params=q_variables["params"],
        tx=optax.adam(learning_rate=config.q_learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2),
        target_params=q_variables["params"]
    )

    return q_state


def sac_state_factory(config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array) -> SACModelState:
    rng_gen = rng_seq(rng_key=rng_key)
    policy_state = create_policy_state(env=env, policy=policy, config=config, rng_key=next(rng_gen))
    q1_state = create_q_state(env, config, next(rng_gen))
    q2_state = create_q_state(env, config, next(rng_gen))

    # although alpha is a scalar, it needs to have a dimension for jax.grad to be happy
    alpha_params = {"alpha": jax.tree_map(lambda alpha: jnp.array([alpha]), config.init_alpha)}

    return SACModelState(policy_state=policy_state,
                         q1_state=q1_state,
                         q2_state=q2_state,
                         alpha_params=alpha_params,
                         alpha_optimizer_params=optax.adam(learning_rate=config.alpha_lr).init(alpha_params),
                         model_clock=jnp.array(0, dtype=jnp.int32), )


def policy_factory(env: gym.Env) -> Policy:
    return Policy(env.action_space)


def q_factory(env: gym.Env) -> QFunction:
    return QFunction(env.action_space)
