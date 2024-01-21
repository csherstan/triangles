# Copyright Craig Sherstan 2024
"""
Contains the implementation of asynchronous SAC.

TODO: Make eval use predictable seeds
TODO: ExpConfig has to be all jax types, otherwise it can't be passed to the jit functions.
"""
import argparse
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from triangles.types import (
    AlphaType,
    PolicyReturnType,
    PolicyType,
    MetricsType,
    MetricWriter,
    NestedNPArray,
    NestedArray,
)
from triangles.util import rng_seq, as_float32, atleast_2d, space_to_reverb_spec


import dataclasses
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Queue, Event, Process
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from queue import Empty
from typing import (
    Optional,
    Any,
    Dict,
    Callable,
    Protocol,
    Tuple,
    List,
    Sequence,
    Mapping,
    cast,
)

import flax.serialization
import flax.struct
import gymnasium as gym
import jax
import numpy as np
import optax
import reverb
import structlog
from flax import struct
from flax.core import FrozenDict
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints
from flax.training.checkpoints import PyTree
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax import Array, numpy as jnp, jit
from reverb import ReplaySample
import tensorflow as tf

LOG = structlog.getLogger()

class QueueMetricWriter(MetricWriter):
    """
    A metric writer that uses a multiprocessing Queue for passing data.
    """

    def __init__(self, queue: Queue):
        self.queue = queue

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.queue.put({"tag": tag, "value": value, "step": step})


class Batch(struct.PyTreeNode):
    obs: PyTree  # o_t
    action: PyTree  # a_{t}
    reward: jnp.ndarray  # r_{t+1}
    terminated: jnp.ndarray  # t_{t+1}
    next_obs: PyTree  # o_{t+1}


def convert_batch(batch: ReplaySample) -> Batch:
    """
    Converts a batch read from the replay buffer into the Batch container for use by the algorithm
    :param batch: Replay sample from reverb buffer.
    :return: Batch object
    """
    return Batch(**jax.tree_map(lambda leaf: atleast_2d(jnp.asarray(leaf)), batch.data))


class FilteredModelQueue:
    """
    For use in sending models to client processes.
    Each incoming model has a model_clock, if `interval` is specified, then
    only models of the corresponding model clock interval are added to the queue.
    """

    def __init__(self, interval: int = 1):
        """
        :param interval: If 1 (default), no filtering is applied
        """
        assert interval >= 1
        self.queue: multiprocessing.Queue = Queue()
        self._interval = interval

    def put(self, model_clock: int, obj: Any) -> None:
        if model_clock % self._interval == 0:
            self.queue.put(obj)


@dataclasses.dataclass(frozen=True)
class RWConfig:
    reverb_address: str
    reverb_table_name: str


@flax.struct.dataclass
class ExpConfig:
    max_replay_size: int = int(1e6)  # max size of the step reverb buffer
    min_replay_size: int = (
        256  # minimum number of steps in the reverb buffer before training begins
    )
    num_rw_workers: int = (
        5  # number of rollout worker process to use for trajectory collection
    )
    seed: Optional[int] = None  # seed for random num gen
    q_learning_rate: float = 0.001  # learning rate used for the q-function optimizer
    policy_learning_rate: float = 0.0001  # learning rate for the policy optimizer
    batch_size: int = 256  # batch size to draw on each training step
    gamma: float = 0.99  # discount value used for TD bootstrapping
    tau: float = 0.995  # soft-target update param, target = target*tau + active*(1-tau)
    init_alpha: float | Mapping[str, float] = 0.5  # weight on the entropy term
    alpha_lr: float = 3e-4  # original code base default
    num_eval_iterations: int = 3  # for each model clock eval, how many episodes to run
    eval_interval: int = 100  # how often, in model_clocks to perform an evaluation
    checkpoint_interval: int = 100  # how often to keep checkpoints


class PolicyFactory(Protocol):
    def __call__(self, env: gym.Env) -> PolicyType:
        pass


class EnvFactory(Protocol):
    def __call__(self, show: bool = False) -> gym.Env:
        pass


class PolicyTrainState(train_state.TrainState):
    """
    Overriding just I can type the apply_fn
    """

    apply_fn: PolicyType = struct.field(pytree_node=False)


class QTrainState(train_state.TrainState):
    """
    This is the train state for the Q-functions. Rather than keep a separate state for the Q-target parameters
    I've just added them here. They operate on the same Q-function and no optimizer is required.
    """

    target_params: FrozenDict[str, Any] = struct.field(pytree_node=True)


class SACModelState(struct.PyTreeNode):
    """
    Holds all the state for all the models and parameters used by SAC
    """

    policy_state: PolicyTrainState
    q1_state: QTrainState
    q2_state: QTrainState

    # I think the cleanest way to wrap up alpha would be to put it in a TrainState as well, but I'm
    # making the choice not to so that I have practice manually applying the transformations and tracking state
    alpha_params: Mapping[str, PyTree]
    alpha_optimizer_params: optax.GradientTransformation

    model_clock: jax.Array  # the model clock (number of training steps) associated with the state


@dataclasses.dataclass()
class TransitionStep:
    """
    Dataclass for capturing a single transition step
    """

    obs: NestedNPArray  # Starting state, s_t
    action: NestedNPArray  # Action taken from s_t->a_t
    reward: np.ndarray  # Reward received from starting in s_t, taking a_t and arriving in state s_{t+1}
    terminated: np.ndarray  # The episode has been terminated because s_{t+1} is a terminal state
    truncated: np.ndarray  # The episode was terminated, but s_{t+1} is not terminal, s_{t+1} can be used for bootstrapping
    info: Dict[str, Any]  # Additional step info


def collect(
    env: gym.Env,
    policy: PolicyType,
    policy_params: VariableDict,
    rng_key: Array,
    exploit: bool = False,
) -> Tuple[float, List[TransitionStep]]:
    """
    Runs a policy for one episode.

    I'm making a constraint that the policy generate actions in [-1, 1]. Those raw actions are what will be
    saved in the replay buffer and any scaling that needs to be done will happen before being sent the environment

    :param env: Environment
    :param policy: The policy nn.Module
    :param policy_params: policy params
    :param rng_key: random key
    :param exploit: if set to True, will use the deterministic output of the policy rather than the sampled ones.
    :return: Tuple[the return, the trajectory as a list of TransitionStep]
    """

    # It would probably be better to just add some sort of preprocessor system.
    def convert_action(action_space: gym.spaces.Space, action: Any) -> NestedNPArray:
        if isinstance(action_space, spaces.Dict):
            return {k: convert_action(v, action[k]) for k, v in action_space.items()}

        if isinstance(action_space, spaces.Discrete):
            assert isinstance(action, (jnp.ndarray, np.ndarray))
            return np.array(action[0], dtype=np.int64)

        if isinstance(action_space, spaces.Box):
            action = np.clip(as_float32(action), a_min=-1.0, a_max=1.0)[0]
            return np.array(
                (
                    action * np.abs(action_space.high - action_space.low) / 2
                    + action_space.low
                    + 1
                )
            )

        raise Exception

    rng_gen = rng_seq(rng_key=rng_key)
    the_return = 0.0
    obs, _ = env.reset(seed=next(rng_gen)[0].item())
    transitions = []
    while True:
        policy_result = policy(
            {"params": policy_params}, jnp.asarray(obs), next(rng_gen)
        )
        action = (
            policy_result.deterministic_actions
            if exploit
            else policy_result.sampled_actions
        )
        np_action = convert_action(env.action_space, action)
        next_obs, reward, terminated, truncated, info = env.step(np_action)
        reward = float(reward)
        transitions.append(
            TransitionStep(
                obs=obs,
                action=np_action,
                reward=as_float32(reward),
                terminated=as_float32(terminated),
                truncated=as_float32(truncated),
                info=info,
            )
        )
        the_return += reward
        obs = next_obs

        if terminated or truncated:
            transitions.append(
                TransitionStep(
                    obs=obs,
                    action=np_action,
                    reward=as_float32(reward),
                    terminated=as_float32(terminated),
                    truncated=as_float32(truncated),
                    info=info,
                )
            )
            break

    return the_return, transitions


def eval_process(
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    model_queue: multiprocessing.Queue,
    metric_queue: multiprocessing.Queue,
    config: ExpConfig,
    rng_key: Array,
) -> None:

    """
    Eval process entrypoint. All jax ops are done on CPU.\

    :param shutdown: Event to watch for a shutdown message
    :param env_factory: makes and env
    :param policy_factory: makes a policy
    :param model_queue: use to receive models
    :param metric_queue: use to write metrics
    :param config: Exp config
    :param rng_key: random key
    :return: None
    """

    try:
        with jax.default_device(jax.devices("cpu")[0]):
            rng_gen = rng_seq(rng_key=rng_key)
            metric_writer = QueueMetricWriter(metric_queue)

            while not shutdown.is_set():
                try:
                    # try to get a new model, if we can't get one it throws an exception and we try the loop again.
                    data = model_queue.get(timeout=1)

                    model_data = deserialize_model(data)
                    params = model_data["policy_params"]
                    model_clock = model_data["model_clock"]
                    LOG.info(f"Eval received {model_clock}")

                    # only evaluate if this is a model_clock at a valid interval
                    if model_clock % config.eval_interval == 0:
                        LOG.info(f"Starting eval for model_clock: {model_clock}")
                        start_time = time.time()
                        eval_step(
                            env_factory=env_factory,
                            policy_factory=policy_factory,
                            policy_params=params,
                            model_clock=model_clock,
                            num_eval_iterations=config.num_eval_iterations,
                            rng_key=next(rng_gen),
                            metric_writer=metric_writer,
                        )
                        LOG.info(
                            f"Completed eval for model_clock ({model_clock}) in {time.time() - start_time} s"
                        )

                except Empty:
                    pass
    except Exception as e:
        LOG.info(e)


def eval_step(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    policy_params: VariableDict,
    model_clock: int,
    num_eval_iterations: int,
    rng_key: Array,
    metric_writer: MetricWriter,
) -> None:

    """
    Collects one or more eval trajectories. An eval trajectory uses the deterministic actions from the policy
    rather than the stochastic ones.

    :param env_factory: Make as an env
    :param policy_factory: Makes a policy
    :param policy_params: policy model params
    :param model_clock: model clock associated with the current policy params
    :param num_eval_iterations: number of evals to run on this model clock
    :param rng_key: random key
    :param metric_writer: write metrics here
    :return: None
    """

    rng_gen = rng_seq(rng_key=rng_key)
    env = env_factory()
    policy = policy_factory(env)

    returns = []
    episode_length = []
    episode_time = []
    step_time = []
    for i in range(num_eval_iterations):
        start_time = time.time()
        the_return, transitions = collect(
            env, policy, policy_params, rng_key=next(rng_gen), exploit=True
        )

        # metrics
        delta_time = time.time() - start_time
        returns.append(the_return)
        episode_length.append(len(transitions) - 1)
        episode_time.append(delta_time)
        step_time.append(delta_time / len(transitions))

    # TODO: these metrics could be handled more cleanly. Could write a metric tracker... maybe something like that is
    # in flax or something already.
    metric_writer.scalar("eval_return", float(np.mean(returns)), model_clock)
    metric_writer.scalar("eval_ep_length", float(np.mean(episode_length)), model_clock)
    metric_writer.scalar("eval_ep_time", float(np.mean(episode_time)), model_clock)
    metric_writer.scalar("eval_step_time", float(np.mean(step_time)), model_clock)


def rollout_worker_fn(
    rw_id: int,
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    queue: multiprocessing.Queue,
    config: RWConfig,
) -> None:
    """
    A wrapper for the rollout worker process. It just makes sure that jax uses the CPU for the rollout worker process.
    :param rw_id: Id to use for logging.
    :param shutdown: An event to monitor for a shutdown event.
    :param env_factory: Call to generate an environment
    :param policy_factory: Call to generate a policy
    :param queue: Queue used for receiving models from the trainer.
    :param config: Experiment config
    :return: None
    """
    with jax.default_device(jax.devices("cpu")[0]):
        assert jnp.array([0]).devices().pop().platform == "cpu"
        rollout_worker(rw_id, shutdown, env_factory, policy_factory, queue, config)


def rollout_worker(
    rw_id: int,
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    model_queue: Queue,
    config: RWConfig,
) -> None:
    """
    A loop to constantly receive updated models from the trainer and run them in exploration mode.
    Trajectories are sent back to the trainer via reverb.

    :param rw_id: Id to use for logging.
    :param shutdown: An event to monitor for a shutdown event.
    :param env_factory: Call to generate an environment
    :param policy_factory: Call to generate a policy
    :param model_queue: Queue used for receiving models from the trainer.
    :param config: Experiment config
    :return: None
    """

    def get_latest(current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gets the latest model from the model_queue. Only keeps the most current model, discards all others.
        Returns the current model if there is no newer one available.

        :param current: Current model params
        :return: the model to use. This will be model with highest model_clock
        """
        result = None
        while result is None:
            received = None
            try:
                while True:
                    received = model_queue.get(timeout=0.1)
            except Empty:
                pass
            result = (
                deserialize_model(received=received)
                if received is not None
                else current
            )

            # This should probably never happen.
            if result is None:
                time.sleep(0.1)

        return result

    LOG.info(f"rw{rw_id} process starting")

    # setup
    reverb_client = reverb.Client(config.reverb_address)
    env = env_factory()
    policy = policy_factory(env)
    rng_gen = rng_seq(seed=time.time_ns())

    # loop until we get a shutdown signal
    rw_model = None
    while not shutdown.is_set():
        rw_model = get_latest(rw_model)

        model_clock = rw_model["model_clock"]
        policy_params = rw_model["policy_params"]
        LOG.info(f"{rw_id} Received model {model_clock}")

        the_return, trajectory = collect(env, policy, policy_params, next(rng_gen))

        write_trajectory(reverb_client, config.reverb_table_name, trajectory)

    LOG.info(f"{rw_id} shutting down rw")


def write_trajectory(
    reverb_client: reverb.Client,
    reverb_table_name: str,
    trajectory: List[TransitionStep],
) -> None:

    """
    Writes a trajectory to the reverb replay buffer. Currently only supports 1-step trajectories.

    :param reverb_client: Reverb client
    :param reverb_table_name: Table name
    :param trajectory: The complete trajectory
    :return: None
    """

    with reverb_client.trajectory_writer(num_keep_alive_refs=2) as writer:
        for idx, step in enumerate(trajectory):
            writer.append(dataclasses.asdict(step))
            if idx > 0:
                try:
                    """
                    Note to self:
                    - this did not behave as expected. So `num_keep_alive_refs` above specifies the size of the
                    revolving buffer. Here, I've set it to 2. However, indexing does not work as expected.

                    Let's say I have a trajectory of length 2, named 'reward': [3, 4,].
                    If I do:
                        - writer.history["reward"][0].numpy() -> 3
                        - writer.history["reward"][1].numpy() -> 4
                        - writer.history["reward"][2].numpy() -> 3
                    So it allows me to access the wrap around value.

                    Next, even though the buffer is size 2 I have to continue to advance my indices. So if I have
                    a trajectory of length 3, named 'reward': [3, 4, 5].

                    First create_item call:
                        - writer.history["reward"][0].numpy() -> 3
                        - writer.history["reward"][1].numpy() -> 4
                    Second create_item call:
                        - writer.history["reward"][1].numpy() -> 4
                        - writer.history["reward"][2].numpy() -> 5

                    """
                    writer.create_item(
                        table=reverb_table_name,
                        trajectory={
                            "obs": jax.tree_map(
                                lambda data: data[idx - 1], writer.history["obs"]
                            ),
                            "action": jax.tree_map(
                                lambda data: data[idx - 1], writer.history["action"]
                            ),
                            "reward": jax.tree_map(
                                lambda data: data[idx - 1], writer.history["reward"]
                            ),
                            "terminated": jax.tree_map(
                                lambda data: data[idx - 1], writer.history["terminated"]
                            ),
                            "next_obs": jax.tree_map(
                                lambda data: data[idx], writer.history["obs"]
                            ),
                        },
                        priority=1,
                    )
                except Exception as e:
                    raise e
            writer.flush()


def calculate_alpha(alpha_params: AlphaType) -> NestedArray:
    """
    Calculates `alpha`, the weight applied to the entropy bonus.
    :param alpha_params: The logits of alpha.
    :return: Computed weights in the same structure as alpha_params[`alpha`]
    """
    return cast(NestedArray, jax.tree_map(lambda v: jnp.exp(v), alpha_params["alpha"]))


def compute_entropy_bonus(alpha_params: AlphaType, logits: NestedArray) -> Array:
    """
    Compute the entropy bonus.
    :param alpha_params: the weight applied to the entropy bonus, this can be a PyTree.
    :param logits: log probability. structure must match alpha unless alpha is a float Also a PyTree
    :return: The summation of the entropy bonus. NOTE: to treat this as a bonus, the negative is applied here.
    The greater the entropy the more positive this value will be.
    """

    alpha_tree = calculate_alpha(alpha_params)

    entropy_bonus_tree = jax.tree_map(
        lambda alpha, logits: -alpha * logits, alpha_tree, logits
    )
    return cast(
        Array,
        jax.tree_util.tree_reduce(
            lambda accumulated, num: accumulated + num, entropy_bonus_tree
        ),
    )


@jit
def q_function_update(
    batch: Batch,
    gamma: float,
    alpha_params: AlphaType,
    tau: float,
    policy_state: PolicyTrainState,
    q1_state: QTrainState,
    q2_state: QTrainState,
    rng_key: Array,
) -> Tuple[Tuple[QTrainState, QTrainState], MetricsType]:

    """
    Qfunction update:

    Uses TD 1-step target, where the bootstrap value is the min \tilde{Q}(s_{t+1}, a_{t+1}) between the 2
    different TARGET \tilde{Q} functions.

    \delta_t = r_t + \gamma (1-done) \tilde{Q}_min(s_{t+1}, \pi(s_{t+1})) - Q(s_t, a_t).
    Additionally, the entropy bonus is added. The entropy bonus is: -log pi(a_{t+1}|s_{t+1}).
    Here: a_t, s_t, s_{t+1} all come from the batch data, but the action, a_{t+1} in the Q-target is sampled from
    the policy (outside the differentiation).

    :param batch: transitions batch data
    :param gamma: bootstrap value [0, 1)
    :param alpha_params: weight applied to the entropy bonus
    :param tau: [0, 1] the rate of update applied to the target Q-networks. Determines what percentage of the target
    network to keep. A value of 0.9 keeps 90% of the target weights and 10% of active Q-functions weights.
    :param policy_state: Policy model params
    :param q1_state: q1 model params
    :param q2_state: q2 model params
    :param rng_key: random key
    :return: Tuple[Tuple[q1_params, q2_params], metrics]
    """

    rng_gen = rng_seq(rng_key=rng_key)
    metrics: Dict[str, Any] = {}

    # sample actions a_{t+1}
    next_state_policy_result = policy_state.apply_fn(
        {"params": policy_state.params}, batch.next_obs, next(rng_gen)
    )

    # get the Q-target value estimates for \tilde{Q}(s_{t+1},a_{t+1})
    target_values_1: Array = q1_state.apply_fn(
        {"params": q1_state.target_params},
        batch.next_obs,
        next_state_policy_result.sampled_actions,
    )
    target_values_2: Array = q2_state.apply_fn(
        {"params": q2_state.target_params},
        batch.next_obs,
        next_state_policy_result.sampled_actions,
    )

    # this handles single action spaces or dictionary action spaces
    entropy_bonus = compute_entropy_bonus(
        alpha_params, next_state_policy_result.log_probabilities
    )

    # r_{t+1} + \gamma * (1-done) * Q_min(s_{t+1}, a_{t+1}~\pi(s_{t+1}) - entropy bonus
    prediction_target = batch.reward + gamma * (1 - batch.terminated) * (
        jnp.minimum(target_values_1, target_values_2) + entropy_bonus
    )

    # Note to self: by default, value_and_grad will take the derivative of the loss (first returned val) wrt the first
    # param, so the params that we want grads for need to be the first argument. This can be changed though.
    def q_loss_fn(
        q_state_params: VariableDict,  # q function params that will get updated
        q_state: TrainState,  # q-function state
        prediction_target: Array,  # we want our predictor to predict the expectatin of this value.
        observations: Array,  # s_t
        actions: Array,  # a_{t+1}
    ) -> Array:
        predicted_q = q_state.apply_fn(
            {"params": q_state_params}, observations, actions
        )
        return jnp.mean(jnp.square(predicted_q - prediction_target))

    q_grad_fn = jax.value_and_grad(q_loss_fn, has_aux=False)
    q1_loss, grads = q_grad_fn(
        q1_state.params, q1_state, prediction_target, batch.obs, batch.action
    )
    q1_state = q1_state.apply_gradients(grads=grads)
    q2_loss, grads = q_grad_fn(
        q2_state.params, q2_state, prediction_target, batch.obs, batch.action
    )
    q2_state = q2_state.apply_gradients(grads=grads)
    metrics["loss.q"] = (q1_loss + q2_loss) / 2.0

    # move the target networks a small amount towards the active networks.
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
    alpha_params: AlphaType,
    policy_state: PolicyTrainState,
    q1_state: QTrainState,
    q2_state: QTrainState,
    rng_key: Array,
) -> Tuple[TrainState, MetricsType]:
    """

    Updates the policy params.

    The objective is:

    J = E[Q_min(s_t,a_t) - \alpha log(\pi(a_t|s_t))] where Q_min is the min value between the two different
    Q functions, alpha is a weight on the entry bonus, given by the log probability.

    :param batch: batch of transiton data
    :param alpha_params: parameters for the weight on the entropy bonus
    :param policy_state: policy model params
    :param q1_state: Qfunction 1 model params
    :param q2_state: Qfunction 2 model params
    :param rng_key: random key
    :return: Tuple[new policy params, metrics]
    """

    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    def policy_loss_fn(
        policy_params: VariableDict, observations: Array, rng_key: Array
    ) -> Array:
        policy_result = policy_state.apply_fn(
            {"params": policy_params}, observations, rng_key
        )

        # compute q values through both Q networks
        q_1 = q1_state.apply_fn(
            {"params": q1_state.params}, observations, policy_result.sampled_actions
        )
        q_2 = q2_state.apply_fn(
            {"params": q2_state.params}, observations, policy_result.sampled_actions
        )

        # Take the sample-wise minimum of the two predictions
        min_q = jnp.minimum(q_1, q_2)

        # add the entropy bonus
        entropy_bonus = compute_entropy_bonus(
            alpha_params, policy_result.log_probabilities
        )
        loss = -jnp.mean(entropy_bonus + min_q)

        return loss

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
    policy_loss, grads = policy_grad_fn(policy_state.params, batch.obs, next(rng_gen))
    policy_state = policy_state.apply_gradients(grads=grads)
    metrics["loss.policy"] = policy_loss

    return policy_state, metrics


@jit
def alpha_update(
    batch: Batch,
    policy_state: PolicyTrainState,
    target_entropy: AlphaType,
    alpha_params: AlphaType,
    alpha_lr: float,
    alpha_optimizer_params: optax.GradientTransformation,
    rng_key: Array,
) -> Tuple[Tuple[AlphaType, VariableDict], MetricsType]:
    """
    Update the alpha parameter: the weight on the entropy bonus.

    Loss: -\alpha E[log_prob(s,a) + target_entropy]. So the result is that when the entropy is higher than the target
    (log_prob is < target_entropy) then we reduce alpha, and when the entropy is lower than the target (our policy
    is collapsing: log_prob < target_entropy) then we increase alpha

    :param batch: The data batch
    :param policy_state:    policy parameters
    :param target_entropy:  target value for the entropy
    :param alpha_params:    current alpha parameters
    :param alpha_lr:    learning rate for the alpha param
    :param alpha_optimizer_params:  alpha optimizer params
    :param rng_key: random key
    :return: Tuple[Tuple[alpha params, alpha_optimizer_params], metrics]
    """

    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    # Sample actions/log_probs for state S
    policy_result = policy_state.apply_fn(
        {"params": policy_state.params}, batch.obs, next(rng_gen)
    )

    def alpha_loss_fn(alpha_params: AlphaType) -> Array:
        # tree_map allows handling lists/dicts/scalar vals of alpha params.
        # this is useful when are splitting up alpha over different parts of the action space as
        # we might do with some approaches to mixed action spaces.
        element_losses = jax.tree_map(
            lambda alpha, log_p, target: -alpha * (log_p + target),
            calculate_alpha(alpha_params),
            policy_result.log_probabilities,
            target_entropy,
        )
        return jnp.array(jax.tree_util.tree_flatten(element_losses)[0]).mean()

    # TODO: in the case of a Dict action space it would be far more useful to keep the losses separate for metrics
    alpha_loss, grads = jax.value_and_grad(alpha_loss_fn, has_aux=False)(alpha_params)
    updates, alpha_optimizer_params = optax.adam(learning_rate=alpha_lr).update(
        grads, alpha_optimizer_params, alpha_params
    )
    alpha_params = optax.apply_updates(alpha_params, updates)

    metrics["loss.alpha"] = alpha_loss

    alphas = calculate_alpha(alpha_params)

    if isinstance(alphas, dict):
        for k, v in alphas.items():
            metrics[f"alpha.{k}"] = v[0]
    else:
        metrics["alpha"] = alphas

    return (alpha_params, alpha_optimizer_params), metrics


def train_step(
    action_space: spaces.Space,
    batch: Batch,
    model_state: SACModelState,
    config: ExpConfig,
    rng_key: Array,
) -> Tuple[SACModelState, Dict[str, float]]:
    """
    Things to watch for:
    - silent broadcasting.
    - min/max operations that reduce when you expect them to be elementwise.
    - accessing elements that can't be traced by jax.

    :param action_space: Action space of the environment
    :param batch:   batch of trajectories
    :param model_state: model/alg params
    :param config:  Experiment config
    :param rng_key: random key
    :return: Tuple[Model state, metrics]
    """

    rng_gen = rng_seq(rng_key=rng_key)

    metrics = {}

    policy_state = model_state.policy_state
    q1_state = model_state.q1_state
    q2_state = model_state.q2_state

    (q1_state, q2_state), q_metrics = q_function_update(
        batch=batch,
        gamma=config.gamma,
        alpha_params=model_state.alpha_params,
        tau=config.tau,
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        rng_key=next(rng_gen),
    )
    metrics.update(q_metrics)

    policy_state, policy_metrics = policy_update(
        batch=batch,
        alpha_params=model_state.alpha_params,
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        rng_key=next(rng_gen),
    )
    metrics.update(policy_metrics)

    def compute_target_entropy(action_space: spaces.Space) -> float | NestedArray:
        """
        Heuristic used in the original paper and codebase (at least for the continuous space).
        Just -1*the product of the space. So for a continuous-space of length 3, this is just -3.
        :param action_space:
        :return: Alpha target entropy
        """
        if isinstance(action_space, spaces.Box):
            return -jnp.prod(jnp.array(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            return -float(action_space.n)
        elif isinstance(action_space, spaces.Dict):
            # weird typing error that I haven't resolved.
            return {k: compute_target_entropy(v) for k, v in action_space.items()}  # type: ignore

        raise Exception(f"Unsupported space {type(action_space)}")

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
    # referencing `model_state.model_clock` I was accessing `sac_state.model_clock`, but sac_state was a global var,
    # therefore it wasn't being traced and the value of the model clock was being cached on the first pass.
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


class SACStateFactory(Protocol):
    """
    Protocol defines the expected call signature used to generate a SACModelState object
    """

    def __call__(
        self, config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array
    ) -> SACModelState:
        pass


def serialize_model(policy_state: PolicyTrainState, model_clock: int) -> bytes:
    return flax.serialization.msgpack_serialize(
        {"policy_params": policy_state.params, "model_clock": model_clock}
    )


def deserialize_model(received: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], flax.serialization.msgpack_restore(bytearray(received)))


def train_loop(
    name: str,
    config: ExpConfig,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    sac_state_factory: SACStateFactory,
) -> None:
    """
    Entrypoint for running the training loop:
    - A trainer processes
    - Multiple worker processes used for rollout collection and evaluation.
    - communication between the workers is managed by the trainer process.
    - Communication of trajectory information is handled by a reverb server owned by the trainer process.
    - Communication of metrics is done via a multiprocessing Queue
    - After each train step the trainer sends the newest model out to the worker processes.
    - One downside of the approach implemented here: evals are run at intervals of model clocks, ex. every 100 training
    steps. However, if the time taken by the eval process is long than the time taken to generate the next eval's model
    clock, then the eval process will never catch up. Three ways we could resolve this: 1. Record collection trajectory
    returns instead (this doesn't really give us the data we want). 2. Launch evals as a completely separate process.
    3. Pause training and rollout collection while evals are running. My preference is option 2 - use a separate process

    Note: moving everything into a main function, instead of leaving it tucked under if __name__=="__main__", can
    address two problems:
      - It can be written in such a way as to be a reusable entry point, callable from different scripts (not done here)
      - It prevents variables from being exposed as global. Global vars caused at least one problem for me here when
      using autocomplete in my IDE.

    :param name: the name to use when writing output artifacts
    :param config: Experiment configuration
    :param env_factory: makes an environment
    :param policy_factory: makes a policy
    :param sac_state_factory: builds the state information
    :return:
    """

    # have to use "spawn" otherwise there are problems launch multiprocesses due to CUDA.
    multiprocessing.set_start_method("spawn")

    # Trainer is the only process that is going to run on the GPU
    assert jnp.array([0]).devices().pop().platform == "gpu"

    seed = config.seed if config.seed is not None else time.time_ns()
    rng_gen = rng_seq(seed=seed)

    env = env_factory()
    policy = policy_factory(env)

    output_dir = Path(
        f"data/{name}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}"
    ).absolute()
    LOG.info(f"Output dir: {output_dir}")

    # ---- metric writing
    summary_writer = SummaryWriter(output_dir / "tensorboard")

    def write_metrics(
        data: Any, model_clock: int, names: Optional[Sequence[str] | str] = None
    ) -> None:
        """
        Write metrics. If this is a nested structure it will use recursion to flatten the name before
        writing the value.

        :param data: The data to write
        :param model_clock: model clock associated with the metric
        :param names: metric name prefix. It will be recursively built for nested metrics, ex. ["loss", "q"]
        :return: None
        """
        names = [] if names is None else [names] if isinstance(names, str) else names
        assert isinstance(names, list)

        # this is a nested structure, flatten first
        if isinstance(data, dict):
            for k, v in data.items():
                write_metrics(data=v, model_clock=model_clock, names=names + [k])
        else:
            metric_name = ".".join(names)
            assert len(metric_name) > 0, "Zero-length metric name."
            summary_writer.scalar(metric_name, float(data), model_clock)

    # ---- initialize Alg state
    sac_state = sac_state_factory(
        config=config, env=env, policy=policy, rng_key=next(rng_gen)
    )

    # ---------- Reverb setup
    reverb_table_name = "table"
    replay_server = reverb.Server(
        tables=[
            reverb.Table(
                name=reverb_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=config.max_replay_size,
                rate_limiter=reverb.rate_limiters.MinSize(config.min_replay_size),
                signature={
                    "obs": space_to_reverb_spec(env.observation_space),
                    "action": space_to_reverb_spec(env.action_space),
                    "reward": tf.TensorSpec((), tf.float32),
                    "terminated": tf.TensorSpec((), tf.float32),
                    "next_obs": space_to_reverb_spec(env.observation_space),
                },
            )
        ]
    )

    reverb_address = f"localhost:{replay_server.port}"

    replay_client = reverb.Client(reverb_address)

    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=reverb_address,
        table=reverb_table_name,
        max_in_flight_samples_per_worker=config.batch_size * 2,
    ).batch(config.batch_size)

    rw_config = RWConfig(
        reverb_address=reverb_address, reverb_table_name=reverb_table_name
    )

    # ------ Set up processes for data collection.
    terminate_event: EventClass = Event()

    model_queues = [FilteredModelQueue() for _ in range(config.num_rw_workers)]
    processes = [
        Process(
            target=rollout_worker_fn,
            kwargs={
                "rw_id": i,
                "shutdown": terminate_event,
                "env_factory": env_factory,
                "policy_factory": policy_factory,
                "queue": model_queues[i].queue,
                "config": rw_config,
            },
        )
        for i in range(config.num_rw_workers)
    ]

    # receives metrics from child processes
    metric_queue: multiprocessing.Queue = multiprocessing.Queue()

    # the eval queue won't send out models that don't meet the specified interval
    eval_model_queue = FilteredModelQueue(interval=config.eval_interval)
    model_queues.append(eval_model_queue)

    # The eval process
    processes.append(
        Process(
            target=eval_process,
            kwargs={
                "shutdown": terminate_event,
                "env_factory": env_factory,
                "policy_factory": policy_factory,
                "model_queue": eval_model_queue.queue,
                "metric_queue": metric_queue,
                "config": config,
                "rng_key": next(rng_gen),
            },
        )
    )
    for p in processes:
        p.start()

    def send_model(model_clock: int) -> None:
        """
        Send model parameters to child processes
        :param model_clock: model clock of the model params
        :return:
        """
        msg = serialize_model(
            policy_state=sac_state.policy_state, model_clock=model_clock
        )
        for q in model_queues:
            q.put(model_clock=model_clock, obj=msg)

    # Send the initial model
    send_model(model_clock=0)

    # ----------- checkpointer

    checkpoint_dir = Path(output_dir / "checkpoint").absolute()
    checkpoint_dir.mkdir(parents=True)

    def save_checkpoint(state: SACModelState, model_clock: int) -> None:
        """
        Save checkpoint. Only saves at the eval_interval
        :param state: Model params, training params
        :param model_clock: model clock associated with the checkpoint
        :return:
        """
        if model_clock % config.eval_interval == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir,
                target={"state": state},
                step=model_clock,
                keep_every_n_steps=config.checkpoint_interval,
            )

    save_checkpoint(sac_state, 0)

    # -------- Training loop
    while (
        replay_client.server_info()[reverb_table_name].current_size
        < config.min_replay_size
    ):
        time.sleep(1)

    LOG.info("minimum replay buffer requirement met, begin training.")
    while True:
        write_metrics(
            data=replay_client.server_info()[reverb_table_name].current_size,
            model_clock=int(sac_state.model_clock),
            names="replay_size",
        )

        # fetch a batch and convert it to the required format for the alg
        batch = convert_batch(list(dataset.take(1))[0])

        sac_state, metrics = train_step(
            action_space=env.action_space,
            batch=batch,
            model_state=sac_state,
            config=config,
            rng_key=next(rng_gen),
        )

        model_clock = int(sac_state.model_clock)

        send_model(model_clock)
        save_checkpoint(sac_state, model_clock)

        # ---- write metrics
        # training step metrics
        write_metrics(data=metrics, model_clock=model_clock)

        try:
            while True:
                metric_data = metric_queue.get(timeout=0.1)
                write_metrics(
                    data=metric_data["value"],
                    model_clock=metric_data["step"],
                    names=metric_data["tag"],
                )
        except Empty:
            pass

        summary_writer.flush()

    # I don't actually have a graceful way to terminate at this point so this code is never reached.

    terminate_event.set()

    for p in processes:
        p.join()

    replay_server.stop()

    print("exit")


def watch(
    config: ExpConfig,
    checkpoint: Path,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    sac_state_factory: SACStateFactory,
) -> None:
    """
    Entry point to view a trained model. Will run the environment in human_render mode.

    :param config: Experiment configuration
    :param checkpoint: path the checkpoint to load, this should be the directory, NOT the checkpoint file.
    The directory will be named `checkpoint_<model_clock>` and contain a `checkpoint` file and a `_METADATA` file

    :param env_factory: Creates the Environment
    :param policy_factory: Create the policy
    :param sac_state_factory: SAC state information
    :return: None
    """

    with jax.default_device(jax.devices("cpu")[0]):
        rng_gen = rng_seq(seed=time.time_ns())
        env = env_factory(show=True)
        policy = policy_factory(env)
        sac_state = sac_state_factory(
            config=config, env=env, policy=policy, rng_key=next(rng_gen)
        )

        sac_state = checkpoints.restore_checkpoint(checkpoint, {"state": sac_state})[
            "state"
        ]

        while True:
            the_return, _ = collect(
                env, policy, sac_state.policy_state.params, next(rng_gen), exploit=False
            )
            print(the_return)


def main(
    name: str,
    config: ExpConfig,
    args: argparse.Namespace,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    sac_state_factory: SACStateFactory,
) -> None:
    match args.mode:
        case "train":
            train_loop(
                name=name,
                config=config,
                env_factory=env_factory,
                policy_factory=policy_factory,
                sac_state_factory=sac_state_factory,
            )

        case "watch":
            assert args.checkpoint
            watch(
                config=config,
                checkpoint=args.checkpoint,
                env_factory=env_factory,
                policy_factory=policy_factory,
                sac_state_factory=sac_state_factory,
            )


def get_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "watch"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="path to checkpoint folder")
    return parser
