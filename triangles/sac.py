import typing
from typing import Tuple, Dict, cast, Protocol, Mapping

import gymnasium as gym
import jax
import optax
import structlog
from flax import struct
from flax.training.checkpoints import PyTree
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from gymnasium import spaces
from jax import jit, Array, numpy as jnp

from triangles.common import Batch, AlphaType, QTrainState, MetricsType, rng_seq, ExpConfig, DictArrayType, PolicyType, \
    ModelState

LOG = structlog.getLogger()


@jit
def q_function_update(
    batch: Batch,
    gamma: float,
    alpha: AlphaType,
    tau: float,
    policy_state: TrainState,
    q1_state: QTrainState,
    q2_state: QTrainState,
    rng_key: Array,
) -> Tuple[Tuple[QTrainState, QTrainState], MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    next_sampled_actions, next_sampled_actions_logits, *_ = policy_state.apply_fn(
        {"params": policy_state.params}, batch.next_obs, next(rng_gen)
    )

    target_values_1: Array = q1_state.apply_fn(
        {"params": q1_state.target_params}, batch.next_obs, next_sampled_actions
    )
    target_values_2: Array = q2_state.apply_fn(
        {"params": q2_state.target_params}, batch.next_obs, next_sampled_actions
    )

    # this handles single action spaces or dictionary action spaces
    entropy_bonus = compute_entropy_bonus(alpha, next_sampled_actions_logits)

    q_target = batch.reward + gamma * (1 - batch.terminated) * (
        jnp.minimum(target_values_1, target_values_2) - entropy_bonus
    )

    # Note to self: by default, value_and_grad will take the derivate of the loss (first returned val) wrt the first
    # param, so the params that we want grads for need to be the first argument.
    def q_loss_fn(
        q_state_params: VariableDict,
        q_state: TrainState,
        q_target: Array,
        states: Array,
        actions: Array,
    ) -> Array:
        predicted_q = q_state.apply_fn({"params": q_state_params}, states, actions)
        return jnp.mean(jnp.square(predicted_q - q_target))

    q_grad_fn = jax.value_and_grad(q_loss_fn, has_aux=False)
    q1_loss, grads = q_grad_fn(
        q1_state.params, q1_state, q_target, batch.obs, batch.action
    )
    q1_state = q1_state.apply_gradients(grads=grads)
    q2_loss, grads = q_grad_fn(
        q2_state.params, q2_state, q_target, batch.obs, batch.action
    )
    q2_state = q2_state.apply_gradients(grads=grads)
    metrics["loss.q"] = (q1_loss + q2_loss) / 2.0

    def update_target_network(q_state: QTrainState) -> QTrainState:
        target_params = jax.tree_map(
            lambda source, target: (1 - tau) * source + tau * target,
            q_state.params,
            q_state.target_params,
        )
        q_state = q_state.replace(target_params=target_params)

        return q_state

    q1_state = update_target_network(q1_state)
    q2_state = update_target_network(q2_state)

    return (q1_state, q2_state), metrics


@jit
def policy_update(
    batch: Batch,
    alpha: AlphaType,
    policy_state: TrainState,
    q1_state: QTrainState,
    q2_state: QTrainState,
    rng_key: Array,
) -> Tuple[TrainState, MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    def policy_loss_fn(
        policy_params: VariableDict, observations: Array, rng_key: Array
    ) -> Array:
        actions, logits, *_ = policy_state.apply_fn(
            {"params": policy_params}, observations, rng_key
        )
        q_1 = q1_state.apply_fn({"params": q1_state.params}, observations, actions)
        q_2 = q2_state.apply_fn({"params": q2_state.params}, observations, actions)

        min_q = jnp.minimum(q_1, q_2)
        entropy_bonus = compute_entropy_bonus(alpha, logits)
        loss = jnp.mean(entropy_bonus - min_q)

        return loss

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
    policy_loss, grads = policy_grad_fn(policy_state.params, batch.obs, next(rng_gen))
    policy_state = policy_state.apply_gradients(grads=grads)
    metrics["loss.policy"] = policy_loss

    return policy_state, metrics


@jit
def alpha_update(
    batch: Batch,
    policy_state: TrainState,
    target_entropy: AlphaType,
    alpha_params: VariableDict,
    alpha_lr: float,
    alpha_optimizer_params: optax.GradientTransformation,
    rng_key: Array,
) -> Tuple[Tuple[VariableDict, VariableDict], MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    actions, log_p_actions, *_ = policy_state.apply_fn(
        {"params": policy_state.params}, batch.obs, next(rng_gen)
    )

    def alpha_loss_fn(alpha_params: VariableDict) -> Array:
        element_losses = jax.tree_map(
            lambda alpha_param, log_p, target: -alpha_param * (log_p + target),
            alpha_params["alpha"],
            log_p_actions,
            target_entropy,
        )
        return jnp.array(jax.tree_util.tree_flatten(element_losses)[0]).mean()

    # TODO: in the case of a Dict action space it would be far more useful to keep the losses separate for metrics
    alpha_loss, grads = jax.value_and_grad(alpha_loss_fn, has_aux=False)(alpha_params)
    updates, alpha_optimizer_params = optax.adam(learning_rate=alpha_lr).update(
        grads, alpha_optimizer_params, alpha_params
    )
    alpha_params = optax.apply_updates(alpha_params, updates)

    # TODO: I'm not sure about this.
    alpha_params = jax.tree_map(lambda v: jnp.maximum(v, 0.01), alpha_params)

    metrics["loss.alpha"] = alpha_loss

    if isinstance(alpha_params["alpha"], dict):
        for k, v in alpha_params["alpha"].items():
            metrics[f"alpha.{k}"] = v[0]
    else:
        metrics["alpha"] = alpha_params["alpha"]

    return (alpha_params, alpha_optimizer_params), metrics

    # heuristic used in the original paper and codebase


def compute_target_entropy(action_space: spaces.Space) -> AlphaType:
    if isinstance(action_space, spaces.Box):
        return -jnp.prod(jnp.array(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return -float(action_space.n)
    elif isinstance(action_space, spaces.Dict):
        return {k: compute_target_entropy(v) for k, v in action_space.items()}

    raise Exception(f"Unsupported space {type(action_space)}")

class SACModelState(ModelState):
    """
    Holds all the state for all the models and parameters used by SAC
    """

    policy_state: TrainState
    q1_state: QTrainState
    q2_state: QTrainState

    # I think the cleanest way to wrap up alpha would be to put it in a TrainState as well, but I'm
    # making the choice not to so that I have practice manually applying the transformations and tracking state
    alpha_params: Mapping[str, PyTree]
    alpha_optimizer_params: optax.GradientTransformation

    model_clock: jax.Array  # the model clock (number of training steps) associated with the state

def train_step(
    action_space: spaces.Space,
    batch: Batch,
    model_state: ModelState,
    config: ExpConfig,
    rng_key: Array,
) -> Tuple[ModelState, Dict[str, float]]:
    """
    Things to watch for:
    - silent broadcasting.
    - min/max operations that reduce when you expect them to be elementwise.


    :param batch:
    :param model_state:
    :param config:
    :param rng_key:
    :return:
    """

    rng_gen = rng_seq(rng_key=rng_key)

    metrics = {}

    model_state = typing.cast(SACModelState, model_state)

    policy_state = model_state.policy_state
    q1_state = model_state.q1_state
    q2_state = model_state.q2_state
    alpha = model_state.alpha_params["alpha"]

    (q1_state, q2_state), q_metrics = q_function_update(
        batch=batch,
        gamma=config.gamma,
        alpha=alpha,
        tau=config.tau,
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        rng_key=next(rng_gen),
    )
    metrics.update(q_metrics)

    policy_state, policy_metrics = policy_update(
        batch=batch,
        alpha=alpha,
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        rng_key=next(rng_gen),
    )
    metrics.update(policy_metrics)

    # alpha_params = model_state.alpha_params
    # alpha_optimizer_params = model_state.alpha_optimizer_params

    target_entropy = compute_target_entropy(action_space)
    (alpha_params, alpha_optimizer_params), alpha_metrics = alpha_update(
        batch=batch,
        policy_state=policy_state,
        target_entropy=target_entropy,
        alpha_params=model_state.alpha_params,
        alpha_lr=config.alpha_lr,
        alpha_optimizer_params=model_state.alpha_optimizer_params,
        rng_key=next(rng_gen),
    )

    metrics.update(alpha_metrics)

    # Note, I ran into a bug here where the model_clock was only getting updated once when the function was jitted.
    # That's a clear sign that there was some side effect happening. The issue turned out to be that instead of
    # referencing `model_state.model_clock` I was accessing `sac_state.model_clock`, which is a global var, therefore
    # it was a global var that wasn't being traced and the value of the model clock was being cached on the first pass.
    return (
        SACModelState(
            model_clock=model_state.model_clock + 1,
            policy_state=policy_state,
            q1_state=q1_state,
            q2_state=q2_state,
            alpha_params=alpha_params,
            alpha_optimizer_params=alpha_optimizer_params,
        ),
        metrics,
    )


def compute_entropy_bonus(alpha: AlphaType, logits: DictArrayType) -> Array:
    entropy_bonus_tree = jax.tree_map(
        lambda weighting, logits: weighting * logits, alpha, logits
    )
    return cast(Array, jax.tree_util.tree_reduce(
        lambda accumulated, num: accumulated + num, entropy_bonus_tree
    ))


class SACStateFactory(Protocol):
    def __call__(
        self, config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array
    ) -> ModelState:
        pass
