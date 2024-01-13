"""
TODO: Make eval use predictable seeds
TODO: Limit range for MixedAction2D
TODO: ExpConfig has to be all jax types, otherwise it can't be passed to the jit functions.
"""

import dataclasses
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Queue, Event, Process
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from queue import Empty
from typing import Optional, Any, Dict, Callable, Protocol, Tuple, List

import flax.serialization
import flax.struct
import gymnasium as gym
import jax
import numpy as np
import optax
import reverb
import structlog
from flax import struct, linen as nn
from flax.core import FrozenDict
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints
from flax.training.checkpoints import PyTree
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax import Array, numpy as jnp, jit
from jax._src.basearray import ArrayLike
from reverb import ReplaySample
import tensorflow as tf

DictArrayType = Dict[str, Array]
LOG = structlog.getLogger()

PolicyType = nn.Module


def rng_seq(*, seed: Optional[int | ArrayLike] = None, rng_key: Optional[Array] = None):
    """
    Create a generator for using the jax rng keys. Not my idea. I saw it elsewhere, but so far I've liked the pattern.
    :param seed: Random Seed
    :param rng_key: Existing key to split
    :return:
    """
    assert seed is not None or rng_key is not None

    if rng_key is None:
        assert seed is not None
        rng_key = jax.random.PRNGKey(seed)

    while True:
        rng_key, sub_key = jax.random.split(rng_key)
        yield sub_key


def as_float32(data: Any) -> np.ndarray:
    """
    Convert data to float32
    :param data: data to convert
    :return: A float32 numpy array
    """
    return np.asarray(data, dtype=np.float32)


class MetricWriter(Protocol):
    """
    A protocol to define an interface for writing metrics
    """

    def scalar(self, tag: str, value: float, step: int):
        pass


class QueueMetricWriter(MetricWriter):

    def __init__(self, queue: Queue):
        self.queue = queue

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.queue.put({"tag": tag, "value": value, "step": step})


def atleast_2d(data: Array) -> Array:
    if len(data.shape) < 2:
        data = jnp.expand_dims(data, -1)

    return data


def convert_batch(batch: ReplaySample) -> Dict[str, jnp.ndarray]:
    return Batch(**jax.tree_map(lambda leaf: atleast_2d(jnp.asarray(leaf)), batch.data))


class FilterQueue():

    def __init__(self, interval: int = -1):
        self.queue = Queue()
        self._interval = interval

    def put(self, model_clock: int, obj: Any):
        if model_clock % self._interval == 0:
            self.queue.put(obj)


@dataclasses.dataclass(frozen=True)
class RWConfig:
    reverb_address: str
    reverb_table_name: str


ParamsType = Dict[str, Any]


@flax.struct.dataclass
class ExpConfig:
    max_replay_size: int = int(1e6)  # max size of the step reverb buffer
    min_replay_size: int = 256  # minimum number of steps in the reverb buffer before training begins
    num_rw_workers: int = 5  #
    seed: Optional[int] = None
    q_learning_rate: float = 0.001
    policy_learning_rate: float = 0.0001
    adam_beta_1: float = 0.5
    adam_beta_2: float = 0.999
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.995  # soft-target update param, target = target*tau + active*(1-tau)
    alpha: float = 0.5  # weight on the entropy term
    alpha_lr: float = 3e-4  # original code base default
    num_eval_iterations: int = 3
    eval_frequency: int = 100  # how often, in model_clocks to perform an evaluation
    checkpoint_frequency: int = 100  # how often to keep checkpoints
    steps_per_model_clock: int = 300


EnvFactoryType = Callable[[Optional[bool]], gym.Env]
PolicyFactoryType = Callable[[gym.Env], nn.Module]


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
    policy_state: TrainState
    q1_state: QTrainState
    q2_state: QTrainState

    # I think the cleanest way to wrap up alpha would be to put it in a TrainState as well, but I'm
    # making the choice not to so that I have practice manually applying the transformations and tracking state
    alpha_params: VariableDict
    alpha_optimizer_params: optax.GradientTransformation

    model_clock: jax.Array  # the model clock (number of training steps) associated with the state


@dataclasses.dataclass()
class TransitionStep:
    """
    Dataclass for capturing a single transition step
    """
    obs: np.ndarray  # Starting state, s_t
    action: np.ndarray  # Action taken from s_t->a_t
    reward: np.ndarray  # Reward received from starting in s_t, taking a_t and arriving in state s_{t+1}
    terminated: np.ndarray  # The episode has been terminated because s_{t+1} is a terminal state
    truncated: np.ndarray  # The episode was terminated, but s_{t+1} is not terminal, s_{t+1} can be used for bootstrapping
    info: Dict[str, Any]  # Additional step info


def collect(env: gym.Env, policy: nn.Module, policy_params: VariableDict, rng_key: Array, exploit=False) -> Tuple[
    float, List[TransitionStep]]:
    """
    I'm making a constraint that the policy generate actions in [-1, 1]. Those raw actions are what will be
    saved in the replay buffer and any scaling that needs to be done will happen before being sent the environment

    :param env:
    :param policy:
    :param policy_params:
    :param rng_key:
    :param exploit:
    :return:
    """

    # It would probably be better to just add some sort of preprocessor system.
    def convert_action(action_space: gym.spaces, action):
        if isinstance(action_space, spaces.Dict):
            return {k: convert_action(v, action[k]) for k, v in action_space.items()}

        if isinstance(action_space, spaces.Discrete):
            return as_float32(action)

        if isinstance(action_space, spaces.Box):
            action = jnp.clip(as_float32(action), a_min=-1.0, a_max=1.0)[0]
            return action * np.abs(action_space.high - action_space.low) / 2 + action_space.low + 1

        raise Exception

    rng_gen = rng_seq(rng_key=rng_key)
    the_return = 0.
    obs, _ = env.reset(seed=next(rng_gen)[0].item())
    transitions = []
    while True:
        action, log_p, exploit_action = policy.apply({"params": policy_params}, jnp.asarray(obs), next(rng_gen))
        action = exploit_action if exploit else action
        action = convert_action(env.action_space, action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)
        transitions.append(
            TransitionStep(obs=obs,
                           action=action,
                           reward=as_float32(reward),
                           terminated=as_float32(terminated),
                           truncated=as_float32(truncated),
                           info=info))
        the_return += reward
        obs = next_obs

        if terminated or truncated:
            transitions.append(
                TransitionStep(obs=obs,
                               action=action,
                               reward=as_float32(reward),
                               terminated=as_float32(terminated),
                               truncated=as_float32(truncated),
                               info=info))
            break

    return the_return, transitions


def space_to_reverb_spec(space: spaces.Space):
    if isinstance(space, spaces.Box):
        return tf.TensorSpec(shape=space.shape, dtype=tf.float32)
    elif isinstance(space, spaces.Discrete):
        # I don't like working with dimensionless vectors because they can lead to hidden broadcast errors, but
        # I want to keep the space comparable to the original environment
        return tf.TensorSpec(shape=(), dtype=tf.int64)
    elif isinstance(space, spaces.Dict):
        return {k: space_to_reverb_spec(v) for k, v in space.items()}


def eval_process(shutdown: EventClass,
                 env_factory: EnvFactoryType,
                 policy_factory: PolicyFactoryType,
                 model_queue: multiprocessing.Queue,
                 metric_queue: multiprocessing.Queue,
                 config: ExpConfig, rng_key):
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            rng_gen = rng_seq(rng_key=rng_key)
            metric_writer = QueueMetricWriter(metric_queue)

            while not shutdown.is_set():
                try:
                    data = model_queue.get(timeout=1)
                    model_data = flax.serialization.msgpack_restore(bytearray(data))
                    params = model_data["policy_params"]
                    model_clock = model_data["model_clock"]
                    LOG.info(f"Eval received {model_clock}")
                    if model_clock % config.eval_frequency == 0:
                        LOG.info(f"Starting eval for model_clock: {model_clock}")
                        start_time = time.time()
                        eval_step(env_factory=env_factory,
                                  policy_factory=policy_factory,
                                  policy_params=params,
                                  model_clock=model_clock,
                                  config=config,
                                  rng_key=next(rng_gen),
                                  metric_writer=metric_writer)
                        LOG.info(f"Completed eval for model_clock ({model_clock}) in {time.time() - start_time} s")

                except Empty:
                    pass
    except Exception as e:
        LOG.info(e)


def eval_step(env_factory: EnvFactoryType,
              policy_factory: PolicyFactoryType,
              policy_params: VariableDict,
              model_clock: int,
              config: ExpConfig,
              rng_key: Array,
              metric_writer: MetricWriter):
    rng_gen = rng_seq(rng_key=rng_key)
    env = env_factory()
    policy = policy_factory(env)

    returns = []
    episode_length = []
    episode_time = []
    step_time = []
    for i in range(config.num_eval_iterations):
        start_time = time.time()
        the_return, transitions = collect(env, policy, policy_params, rng_key=next(rng_gen), exploit=True)

        # metrics
        delta_time = time.time() - start_time
        returns.append(the_return)
        episode_length.append(len(transitions) - 1)
        episode_time.append(delta_time)
        step_time.append(delta_time / len(transitions))

    metric_writer.scalar("eval_return", np.mean(returns), model_clock)
    metric_writer.scalar("eval_ep_length", np.mean(episode_length), model_clock)
    metric_writer.scalar("eval_ep_time", np.mean(episode_time), model_clock)
    metric_writer.scalar("eval_step_time", np.mean(step_time), model_clock)


MetricsType = Dict[str, Any]


def rw(idx: int, shutdown: EventClass, env_factory: EnvFactoryType, policy_factory: PolicyFactoryType, queue,
       config: RWConfig):
    with jax.default_device(jax.devices('cpu')[0]):
        assert jnp.array([0]).devices().pop().platform == "cpu"
        rw_(idx, shutdown, env_factory, policy_factory, queue, config)


def rw_(rw_id: int, shutdown: EventClass, env_factory: EnvFactoryType, policy_factory: PolicyFactoryType, queue: Queue,
        config: RWConfig):
    """
    data collection process.
    :return:
    """

    def get_latest(current):
        result = None
        while result is None:
            received = None
            try:
                while True:
                    received = queue.get(timeout=0.1)
            except Empty:
                pass
            result = flax.serialization.msgpack_restore(bytearray(received)) if received is not None else current

            if result is None:
                time.sleep(0.1)

        return result

    LOG.info(f"rw{rw_id} process starting")
    # os.environ["JAX_PLATFORMS"] = "cpu"
    # with jax.default_device(jax.devices('cpu')[0]):

    reverb_client = reverb.Client(config.reverb_address)
    env = env_factory()

    policy = policy_factory(env)

    rng_gen = rng_seq(seed=time.time_ns())

    rw_model = None
    while not shutdown.is_set():
        rw_model = get_latest(rw_model)

        model_clock = rw_model["model_clock"]
        policy_params = rw_model["policy_params"]
        LOG.info(f"{rw_id} Received model {model_clock}")

        the_return, trajectory = collect(env, policy, policy_params, next(rng_gen))

        write_trajectory(reverb_client, config.reverb_table_name, trajectory)

    LOG.info(f"{rw_id} shutting down rw")


def write_trajectory(reverb_client, reverb_table_name: str, trajectory: List[TransitionStep]) -> None:
    with reverb_client.trajectory_writer(num_keep_alive_refs=2) as writer:
        for idx, step in enumerate(trajectory):
            writer.append(dataclasses.asdict(step))
            if idx > 0:
                first_slice = lambda data: data[idx - 1:idx]
                try:
                    # writer.create_item(table=reverb_table_name,
                    #                    trajectory={
                    #                      "obs": writer.history["obs"][idx - 1],
                    #                      "action": writer.history["action"][idx - 1],
                    #                      "reward": writer.history["reward"][idx - 1],
                    #                      "terminated": writer.history["terminated"][idx - 1],
                    #                      "next_obs": writer.history["obs"][idx],
                    #                    }, priority=1)
                    writer.create_item(table=reverb_table_name,
                                       trajectory={
                                           "obs": slice_leaves(writer.history["obs"], first_slice),
                                           "action": slice_leaves(writer.history["action"], first_slice),
                                           "reward": slice_leaves(writer.history["reward"], first_slice),
                                           "terminated": slice_leaves(writer.history["terminated"], first_slice),
                                           "next_obs": slice_leaves(writer.history["obs"],
                                                                    lambda data: data[idx:] if idx == len(
                                                                        trajectory) - 1 else data[idx:idx + 1]),
                                       }, priority=1)
                except Exception as e:
                    raise e
            writer.flush()


class Batch(struct.PyTreeNode):
    obs: PyTree
    action: PyTree
    reward: jnp.ndarray
    terminated: jnp.ndarray
    next_obs: PyTree


@jit
def q_function_update(batch: Batch, gamma: float, alpha: float, tau: float, policy_state: TrainState,
                      q1_state: QTrainState,
                      q2_state: QTrainState, rng_key: Array) -> Tuple[Tuple[QTrainState, QTrainState], MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    next_sampled_actions, next_sampled_actions_logits, *_ = policy_state.apply_fn(
        {"params": policy_state.params}, batch.next_obs, next(rng_gen))

    target_values_1: Array = q1_state.apply_fn({"params": q1_state.target_params},
                                               batch.next_obs, next_sampled_actions)
    target_values_2: Array = q2_state.apply_fn({"params": q2_state.target_params},
                                               batch.next_obs, next_sampled_actions)

    q_target = batch.reward + gamma * (1 - batch.terminated) * (jnp.minimum(target_values_1,
                                                                            target_values_2) -
                                                                alpha * next_sampled_actions_logits)

    # Note to self: by default, value_and_grad will take the derivate of the loss (first returned val) wrt the first
    # param, so the params that we want grads for need to be the first argument.
    def q_loss_fn(q_state_params: VariableDict, q_state: TrainState, q_target: Array, states: Array, actions: Array):
        predicted_q = q_state.apply_fn({"params": q_state_params}, states, actions)
        return jnp.mean(jnp.square(predicted_q - q_target))

    q_grad_fn = jax.value_and_grad(q_loss_fn, has_aux=False)
    q1_loss, grads = q_grad_fn(q1_state.params, q1_state, q_target, batch.obs, batch.action)
    q1_state = q1_state.apply_gradients(grads=grads)
    q2_loss, grads = q_grad_fn(q2_state.params, q2_state, q_target, batch.obs, batch.action)
    q2_state = q2_state.apply_gradients(grads=grads)
    metrics["loss.q"] = (q1_loss + q2_loss) / 2.

    def update_target_network(q_state: QTrainState) -> QTrainState:
        target_params = jax.tree_map(lambda source, target: (1 - tau) * source + tau * target, q_state.params,
                                     q_state.target_params)
        q_state = q_state.replace(target_params=target_params)

        return q_state

    q1_state = update_target_network(q1_state)
    q2_state = update_target_network(q2_state)

    return (q1_state, q2_state), metrics


@jit
def policy_update(batch: Batch, alpha: float, policy_state: TrainState, q1_state: QTrainState, q2_state: QTrainState,
                  rng_key: Array) -> Tuple[TrainState, MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    def policy_loss_fn(policy_params: VariableDict, observations: Array, rng_key: Array) -> Array:
        actions, logits, *_ = policy_state.apply_fn({"params": policy_params}, observations, rng_key)
        q_1 = q1_state.apply_fn({"params": q1_state.params}, observations, actions)
        q_2 = q2_state.apply_fn({"params": q2_state.params}, observations, actions)

        min_q = jnp.minimum(q_1, q_2)
        loss = jnp.mean(alpha * logits - min_q)

        return loss

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
    policy_loss, grads = policy_grad_fn(policy_state.params, batch.obs, next(rng_gen))
    policy_state = policy_state.apply_gradients(grads=grads)
    metrics["loss.policy"] = policy_loss

    return policy_state, metrics


@jit
def alpha_update(batch: Batch, policy_state: TrainState, alpha_params: VariableDict, alpha_lr: float,
                 alpha_optimizer_params: optax.GradientTransformation, rng_key: Array) -> Tuple[
    Tuple[VariableDict, VariableDict], MetricsType]:
    rng_gen = rng_seq(rng_key=rng_key)
    metrics = {}

    actions, log_p_actions, *_ = policy_state.apply_fn({"params": policy_state.params}, batch.obs, next(rng_gen))

    # heuristic used in the original paper and codebase
    target_entropy = -jnp.prod(jnp.array(actions.shape[1:]))

    alpha_loss, grads = jax.value_and_grad(
        lambda alpha_params: -(alpha_params["alpha"] * (log_p_actions + target_entropy)).mean(), has_aux=False)(
        alpha_params)
    updates, alpha_optimizer_params = optax.adam(learning_rate=alpha_lr).update(grads,
                                                                                alpha_optimizer_params,
                                                                                alpha_params)
    alpha_params = optax.apply_updates(alpha_params, updates)
    metrics["loss.alpha"] = alpha_loss

    metrics["alpha"] = alpha_params["alpha"][0]

    return (alpha_params, alpha_optimizer_params), metrics


@jit
def train_step(batch: Batch, model_state: SACModelState, config: ExpConfig, rng_key: Array) -> Tuple[
    SACModelState, Dict[str, float]]:
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

    policy_state = model_state.policy_state
    q1_state = model_state.q1_state
    q2_state = model_state.q2_state
    alpha = model_state.alpha_params["alpha"][0]

    (q1_state, q2_state), q_metrics = q_function_update(batch=batch,
                                                        gamma=config.gamma,
                                                        alpha=alpha,
                                                        tau=config.tau,
                                                        policy_state=policy_state,
                                                        q1_state=q1_state,
                                                        q2_state=q2_state,
                                                        rng_key=next(rng_gen))
    metrics.update(q_metrics)

    policy_state, policy_metrics = policy_update(batch=batch,
                                                 alpha=alpha,
                                                 policy_state=policy_state,
                                                 q1_state=q1_state,
                                                 q2_state=q2_state,
                                                 rng_key=next(rng_gen))
    metrics.update(policy_metrics)

    # alpha_params = model_state.alpha_params
    # alpha_optimizer_params = model_state.alpha_optimizer_params

    (alpha_params, alpha_optimizer_params), alpha_metrics = alpha_update(batch=batch, policy_state=policy_state,
                                                                         alpha_params=model_state.alpha_params,
                                                                         alpha_lr=config.alpha_lr,
                                                                         alpha_optimizer_params=model_state.alpha_optimizer_params,
                                                                         rng_key=next(rng_gen))

    metrics.update(alpha_metrics)

    # Note, I ran into a bug here where the model_clock was only getting updated once when the function was jitted.
    # That's a clear sign that there was some side effect happening. The issue turned out to be that instead of
    # referencing `model_state.model_clock` I was accessing `sac_state.model_clock`, which is a global var, therefore
    # it was a global var that wasn't being traced and the value of the model clock was being cached on the first pass.
    return SACModelState(
        model_clock=model_state.model_clock + 1,
        policy_state=policy_state,
        q1_state=q1_state,
        q2_state=q2_state,
        alpha_params=alpha_params,
        alpha_optimizer_params=alpha_optimizer_params,
    ), metrics


@dataclasses.dataclass(frozen=True)
class RWModel:
    model_clock: int
    params: Dict[str, Any]
    model_factory: Callable[[], nn.Module]


SACStateFactory = Callable[[ExpConfig, gym.Env, nn.Module, Array], SACModelState]


def train(name: str, config: ExpConfig, env_factory: EnvFactoryType, policy_factory: PolicyFactoryType,
          sac_state_factory: SACStateFactory) -> None:
    """
    Note: moving everything into a main function, instead of leaving it tucked under if __name__=="__main__", can
    address two problems:
      - It can be written in such a way as to be a reusable entry point, callable from different scripts (not done here).
      - It prevents variables from being exposed as global. Global vars caused at least one problem for me here when
      using autocomplete in my IDE.
    """

    multiprocessing.set_start_method("spawn")

    assert jnp.array([0]).devices().pop().platform == "gpu"

    seed = config.seed if config.seed is not None else time.time_ns()
    rng_gen = rng_seq(seed=seed)

    env = env_factory()
    policy = policy_factory(env)

    output_dir = Path(f"data/{name}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}").absolute()
    LOG.info(f"Output dir: {output_dir}")
    summary_writer = SummaryWriter(output_dir / "tensorboard")

    sac_state = sac_state_factory(config=config,
                                  env=env,
                                  policy=policy,
                                  rng_key=next(rng_gen))

    # ---------- Reverb setup
    reverb_table_name = 'table'
    # create reverb server
    replay_server = reverb.Server(tables=[
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
            }
        )
    ])

    reverb_address = f'localhost:{replay_server.port}'

    replay_client = reverb.Client(reverb_address)

    dataset = reverb.TrajectoryDataset.from_table_signature(server_address=reverb_address,
                                                            table=reverb_table_name,
                                                            max_in_flight_samples_per_worker=config.batch_size * 2).batch(
        config.batch_size)

    rw_config = RWConfig(reverb_address=reverb_address, reverb_table_name=reverb_table_name)

    # ------ Set up processes for data collection.
    terminate_event: EventClass = Event()

    model_queues = [FilterQueue() for i in range(config.num_rw_workers)]
    processes = [Process(target=rw, kwargs={"idx": i,
                                            "shutdown": terminate_event,
                                            "env_factory": env_factory,
                                            "policy_factory": policy_factory,
                                            "queue": model_queues[i].queue,
                                            "config": rw_config}) for i in range(config.num_rw_workers)]

    metric_queue = multiprocessing.Queue()
    eval_model_queue = FilterQueue(interval=config.eval_frequency)
    model_queues.append(eval_model_queue)

    processes.append(
        Process(target=eval_process,
                kwargs={"shutdown": terminate_event,
                        "env_factory": env_factory,
                        "policy_factory": policy_factory,
                        "model_queue": eval_model_queue.queue,
                        "metric_queue": metric_queue,
                        "config": config,
                        "rng_key": next(rng_gen)}))
    for p in processes:
        p.start()

    def send_model(model_clock: int) -> None:
        # LOG.info(f"Sending model: {model_clock}")
        msg = flax.serialization.msgpack_serialize(
            {"policy_params": sac_state.policy_state.params, "model_clock": model_clock})
        for q in model_queues:
            q.put(model_clock=model_clock, obj=msg)

    send_model(model_clock=0)

    # ----------- checkpointer

    checkpoint_dir = Path(output_dir / "checkpoint").absolute()
    checkpoint_dir.mkdir(parents=True)

    def save_checkpoint(state: SACModelState, model_clock: int) -> None:
        if model_clock % config.eval_frequency == 0:
            checkpoints.save_checkpoint(checkpoint_dir, target={"state": state}, step=model_clock,
                                        keep_every_n_steps=config.checkpoint_frequency)

    # -------- Training loop
    while replay_client.server_info()[reverb_table_name].current_size < config.min_replay_size:
        time.sleep(1)

    LOG.info("minimum replay buffer requirement met")

    save_checkpoint(sac_state, 0)

    LOG.info("begin training")
    while True:
        summary_writer.scalar("replay_size", replay_client.server_info()[reverb_table_name].current_size,
                              int(sac_state.model_clock))

        batch = convert_batch(list(dataset.take(1))[0])
        new_sac_state, metrics = train_step(batch, sac_state, config=config, rng_key=next(rng_gen))

        sac_state = new_sac_state
        model_clock = int(sac_state.model_clock)

        send_model(model_clock)
        save_checkpoint(sac_state, model_clock)

        # ---- write metrics
        for k, v in metrics.items():
            summary_writer.scalar(k, float(v), model_clock)

        try:
            while True:
                metric_data = metric_queue.get(timeout=0.1)
                summary_writer.scalar(**metric_data)
        except Empty:
            pass

        summary_writer.flush()

    terminate_event.set()

    for p in processes:
        p.join()

    replay_server.stop()

    print("exit")


def watch(config: ExpConfig, checkpoint: Path, env_factory: EnvFactoryType, policy_factory: PolicyFactoryType,
          sac_state_factory: SACStateFactory) -> None:
    with jax.default_device(jax.devices('cpu')[0]):
        rng_gen = rng_seq(seed=time.time_ns())
        env = env_factory(show=True)
        policy = policy_factory(env)
        sac_state = sac_state_factory(config=config,
                                      env=env,
                                      policy=policy,
                                      rng_key=next(rng_gen))

        sac_state = checkpoints.restore_checkpoint(checkpoint, {"state": sac_state})["state"]

        while True:
            the_return, _ = collect(env, policy, sac_state.policy_state.params, next(rng_gen), exploit=False)
            print(the_return)


def main(name: str, config: ExpConfig, args, env_factory: EnvFactoryType, policy_factory: PolicyFactoryType,
         sac_state_factory: SACStateFactory) -> None:
    match args.mode:
        case "train":
            train(name=name, config=config, env_factory=env_factory, policy_factory=policy_factory,
                  sac_state_factory=sac_state_factory)

        case "watch":
            assert args.checkpoint
            watch(config=config, checkpoint=args.checkpoint, env_factory=env_factory, policy_factory=policy_factory,
                  sac_state_factory=sac_state_factory)


def convert_space_to_jnp(space_data: Any) -> Any:
    if isinstance(space_data, dict):
        return {k: convert_space_to_jnp(v) for k, v in space_data.items()}
    if isinstance(space_data, np.ndarray):
        return jnp.array(space_data)

    return jnp.array(space_data)


def stack_dict_jnp(dict_list: List[DictArrayType]):
    ret_data = {}
    for entry in dict_list:
        for k, v in entry.items():
            ret_data.setdefault(k, []).append(v)
    return {k: jnp.array(v) for k, v in ret_data.items()}


def slice_leaves(data: Any, slice_fn: Callable[[Any], Any]) -> Any:
    return jax.tree_map(lambda entry: slice_fn(entry)[0], data)
