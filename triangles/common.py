"""
TODO: Make eval use predictable seeds
TODO: Limit range for MixedAction2D
TODO: ExpConfig has to be all jax types, otherwise it can't be passed to the jit functions.
"""
import argparse
import os

import optax
from flax.training.train_state import TrainState

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

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
    Iterator, Mapping,
)

import flax.serialization
import flax.struct
import gymnasium as gym
import jax
import numpy as np
import reverb
import structlog
from flax import struct, linen as nn
from flax.core import FrozenDict
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints
from flax.training.checkpoints import PyTree
from gymnasium import spaces
from jax import Array, numpy as jnp
from reverb import ReplaySample
import tensorflow as tf

DictArrayType = Dict[str, Array]
LOG = structlog.getLogger()
AlphaType = float | Array | Dict[str, "AlphaType"]

PolicyType = nn.Module


def rng_seq(
    *, seed: Optional[int] = None, rng_key: Optional[Array] = None
) -> Iterator[Array]:
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
        assert rng_key is not None
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

    def scalar(self, tag: str, value: float, step: int) -> None:
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


class Batch(struct.PyTreeNode):
    obs: PyTree
    action: PyTree
    reward: jnp.ndarray
    terminated: jnp.ndarray
    next_obs: PyTree


def convert_batch(batch: ReplaySample) -> Batch:
    return Batch(**jax.tree_map(lambda leaf: atleast_2d(jnp.asarray(leaf)), batch.data))


class FilterQueue:
    def __init__(self, interval: int = -1):
        self.queue: multiprocessing.Queue = Queue()
        self._interval = interval

    def put(self, model_clock: int, obj: Any) -> None:
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
    min_replay_size: int = (
        256  # minimum number of steps in the reverb buffer before training begins
    )
    num_rw_workers: int = 5  #
    seed: Optional[int] = None
    q_learning_rate: float = 0.001
    policy_learning_rate: float = 0.0001
    adam_beta_1: float = 0.5
    adam_beta_2: float = 0.999
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.995  # soft-target update param, target = target*tau + active*(1-tau)
    init_alpha: AlphaType = 0.5  # weight on the entropy term
    alpha_lr: float = 3e-4  # original code base default
    num_eval_iterations: int = 3
    eval_frequency: int = 100  # how often, in model_clocks to perform an evaluation
    checkpoint_frequency: int = 100  # how often to keep checkpoints
    steps_per_model_clock: int = 300


class PolicyFactory(Protocol):
    def __call__(self, env: gym.Env) -> nn.Module:
        pass


class EnvFactory(Protocol):
    def __call__(self, show: bool = False) -> gym.Env:
        pass

class StateFactory(Protocol):

    def __call__(
        self, config: ExpConfig, env: gym.Env, policy: PolicyType, rng_key: Array
    ) -> PyTree:
        pass

class ModelState(struct.PyTreeNode):
    """
    Holds all the state for all the models and parameters used by SAC
    """

    # policy_state: TrainState
    # q1_state: QTrainState
    # q2_state: QTrainState
    #
    # # I think the cleanest way to wrap up alpha would be to put it in a TrainState as well, but I'm
    # # making the choice not to so that I have practice manually applying the transformations and tracking state
    # alpha_params: Mapping[str, PyTree]
    # alpha_optimizer_params: optax.GradientTransformation

    model_clock: jax.Array  # the model clock (number of training steps) associated with the state

class TrainStep(Protocol):

    def __call__(
        action_space: spaces.Space,
        batch: Batch,
        model_state: ModelState,
        config: ExpConfig,
        rng_key: Array,
    ) -> Tuple[ModelState, Dict[str, float]]:
        pass


class QTrainState(train_state.TrainState):
    """
    This is the train state for the Q-functions. Rather than keep a separate state for the Q-target parameters
    I've just added them here. They operate on the same Q-function and no optimizer is required.
    """

    target_params: FrozenDict[str, Any] = struct.field(pytree_node=True)


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


def collect(
    env: gym.Env,
    policy: nn.Module,
    policy_params: VariableDict,
    rng_key: Array,
    exploit: bool = False,
) -> Tuple[float, List[TransitionStep]]:
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
    def convert_action(action_space: gym.spaces.Space, action: Any) -> Any:
        if isinstance(action_space, spaces.Dict):
            return {k: convert_action(v, action[k]) for k, v in action_space.items()}

        if isinstance(action_space, spaces.Discrete):
            assert isinstance(action, (jnp.ndarray, np.ndarray))
            return np.int64(action[0])

        if isinstance(action_space, spaces.Box):
            action = np.clip(as_float32(action), a_min=-1.0, a_max=1.0)[0]
            return (
                action * np.abs(action_space.high - action_space.low) / 2
                + action_space.low
            )

        raise Exception

    rng_gen = rng_seq(rng_key=rng_key)
    the_return = 0.0
    obs, _ = env.reset(seed=next(rng_gen)[0].item())
    transitions = []
    while True:

        obs_jnp = stack_dict_jnp([obs])

        action, log_p, exploit_action = policy.apply(   # type: ignore
            {"params": policy_params}, obs_jnp, next(rng_gen)
        )
        action = exploit_action if exploit else action
        action = convert_action(env.action_space, action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)
        transitions.append(
            TransitionStep(
                obs=obs,
                action=action,
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
                    action=action,
                    reward=as_float32(reward),
                    terminated=as_float32(terminated),
                    truncated=as_float32(truncated),
                    info=info,
                )
            )
            break

    return the_return, transitions


def space_to_reverb_spec(
    space: spaces.Space,
) -> tf.TensorSpec | Dict[str, tf.TensorSpec]:
    if isinstance(space, spaces.Box):
        return tf.TensorSpec(shape=space.shape, dtype=tf.float32)
    elif isinstance(space, spaces.Discrete):
        # I don't like working with dimensionless vectors because they can lead to hidden broadcast errors, but
        # I want to keep the space comparable to the original environment
        return tf.TensorSpec(shape=(1,), dtype=tf.int64)
    elif isinstance(space, spaces.Dict):
        return {k: space_to_reverb_spec(v) for k, v in space.items()}

    raise Exception(f"Unsupported Type {type(space)}")


def eval_process(
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    model_queue: multiprocessing.Queue,
    metric_queue: multiprocessing.Queue,
    config: ExpConfig,
    rng_key: Array,
) -> None:
    try:
        with jax.default_device(jax.devices("cpu")[0]):
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
                        eval_step(
                            env_factory=env_factory,
                            policy_factory=policy_factory,
                            policy_params=params,
                            model_clock=model_clock,
                            config=config,
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
    config: ExpConfig,
    rng_key: Array,
    metric_writer: MetricWriter,
) -> None:
    rng_gen = rng_seq(rng_key=rng_key)
    env = env_factory()
    policy = policy_factory(env)

    returns = []
    episode_length = []
    episode_time = []
    step_time = []
    for i in range(config.num_eval_iterations):
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

    metric_writer.scalar("eval_return", float(np.mean(returns)), model_clock)
    metric_writer.scalar("eval_ep_length", float(np.mean(episode_length)), model_clock)
    metric_writer.scalar("eval_ep_time", float(np.mean(episode_time)), model_clock)
    metric_writer.scalar("eval_step_time", float(np.mean(step_time)), model_clock)


MetricsType = Dict[str, Any]


def rw(
    idx: int,
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    queue: multiprocessing.Queue,
    config: RWConfig,
) -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        assert jnp.array([0]).devices().pop().platform == "cpu"
        rw_(idx, shutdown, env_factory, policy_factory, queue, config)


def rw_(
    rw_id: int,
    shutdown: EventClass,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    queue: Queue,
    config: RWConfig,
) -> None:
    """
    data collection process.
    :return:
    """

    def get_latest(current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = None
        while result is None:
            received = None
            try:
                while True:
                    received = queue.get(timeout=0.1)
            except Empty:
                pass
            result = (
                flax.serialization.msgpack_restore(bytearray(received))
                if received is not None
                else current
            )

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


def write_trajectory(
    reverb_client: reverb.Client,
    reverb_table_name: str,
    trajectory: List[TransitionStep],
) -> None:
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


@dataclasses.dataclass(frozen=True)
class RWModel:
    model_clock: int
    params: Dict[str, Any]
    model_factory: Callable[[], nn.Module]


def train(
    name: str,
    config: ExpConfig,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    state_factory: StateFactory,
    train_step: TrainStep,
) -> None:
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

    output_dir = Path(
        f"data/{name}/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}"
    ).absolute()
    LOG.info(f"Output dir: {output_dir}")
    summary_writer = SummaryWriter(output_dir / "tensorboard")

    state = state_factory(
        config=config, env=env, policy=policy, rng_key=next(rng_gen)
    )

    # ---------- Reverb setup
    reverb_table_name = "table"
    # create reverb server
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

    model_queues = [FilterQueue() for i in range(config.num_rw_workers)]
    processes = [
        Process(
            target=rw,
            kwargs={
                "idx": i,
                "shutdown": terminate_event,
                "env_factory": env_factory,
                "policy_factory": policy_factory,
                "queue": model_queues[i].queue,
                "config": rw_config,
            },
        )
        for i in range(config.num_rw_workers)
    ]

    metric_queue: multiprocessing.Queue = multiprocessing.Queue()
    eval_model_queue = FilterQueue(interval=config.eval_frequency)
    model_queues.append(eval_model_queue)

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
        # LOG.info(f"Sending model: {model_clock}")
        msg = flax.serialization.msgpack_serialize(
            {"policy_params": state.policy_state.params, "model_clock": model_clock}
        )
        for q in model_queues:
            q.put(model_clock=model_clock, obj=msg)

    send_model(model_clock=0)

    # ----------- checkpointer

    checkpoint_dir = Path(output_dir / "checkpoint").absolute()
    checkpoint_dir.mkdir(parents=True)

    def save_checkpoint(state: ModelState, model_clock: int) -> None:
        if model_clock % config.eval_frequency == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir,
                target={"state": state},
                step=model_clock,
                keep_every_n_steps=config.checkpoint_frequency,
            )

    # -------- Training loop
    while (
        replay_client.server_info()[reverb_table_name].current_size
        < config.min_replay_size
    ):
        time.sleep(1)

    LOG.info("minimum replay buffer requirement met")

    save_checkpoint(state, 0)

    LOG.info("begin training")
    while True:
        summary_writer.scalar(
            "replay_size",
            replay_client.server_info()[reverb_table_name].current_size,
            int(state.model_clock),
        )

        batch = convert_batch(list(dataset.take(1))[0])
        new_state, metrics = train_step(
            action_space=env.action_space,
            batch=batch,
            model_state=state,
            config=config,
            rng_key=next(rng_gen),
        )

        state = new_state
        model_clock = int(state.model_clock)

        send_model(model_clock)
        save_checkpoint(state, model_clock)

        def write_metric(data: Any, prefixes: Optional[Sequence[str]] = None) -> None:
            prefixes = prefixes or []
            if isinstance(data, dict):
                for k, v in data.items():
                    write_metric(v, [] + [k])
            else:
                summary_writer.scalar(".".join(prefixes), float(data.item()), model_clock)

        # ---- write metrics
        write_metric(metrics)

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


def watch(
    config: ExpConfig,
    checkpoint: Path,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    state_factory: StateFactory,
) -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        rng_gen = rng_seq(seed=time.time_ns())
        env = env_factory(show=True)
        policy = policy_factory(env)
        state = state_factory(
            config=config, env=env, policy=policy, rng_key=next(rng_gen)
        )

        state = checkpoints.restore_checkpoint(checkpoint, {"state": state})[
            "state"
        ]

        while True:
            the_return, _ = collect(
                env, policy, state.policy_state.params, next(rng_gen), exploit=False
            )
            print(the_return)


def main(
    name: str,
    config: ExpConfig,
    args: argparse.Namespace,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    state_factory: StateFactory,
    train_step: TrainStep,
) -> None:
    match args.mode:
        case "train":
            train(
                name=name,
                config=config,
                env_factory=env_factory,
                policy_factory=policy_factory,
                state_factory=state_factory,
                train_step=train_step,
            )

        case "watch":
            assert args.checkpoint
            watch(
                config=config,
                checkpoint=args.checkpoint,
                env_factory=env_factory,
                policy_factory=policy_factory,
                state_factory=state_factory,
            )


def convert_space_to_jnp(space_data: Any) -> Any:
    if isinstance(space_data, dict):
        return {k: convert_space_to_jnp(v) for k, v in space_data.items()}
    if isinstance(space_data, np.ndarray):
        return jnp.array(space_data)

    return jnp.array(space_data)


def stack_dict_jnp(dict_list: List[DictArrayType]) -> Dict[str, Array]:
    ret_data: Dict[str, List[Array]] = {}
    for entry in dict_list:
        for k, v in entry.items():
            ret_data.setdefault(k, []).append(v)
    return {k: atleast_2d(jnp.array(v)) for k, v in ret_data.items()}


