"""
This is a training script to first train gym agents in the classic control setting to
validate my SAC implementation.

At the time of writing jaxlib 0.4.23 has a bug from the xla prject that prevents it from obeying XLA_PYTHON_CLIENT_PREALLOCATE
Because of this I have downgraded to 0.4.19
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import multiprocessing

from datetime import datetime
from queue import Empty

import flax.serialization
import numpy as np
import optax
from flax.core import FrozenDict
from flax.core.scope import VariableDict
from flax.metrics.tensorboard import SummaryWriter
from jax._src.basearray import ArrayLike
from reverb import ReplaySample

import dataclasses
import time
from typing import Dict, Any, Optional, Tuple, List, Callable, Protocol

import gymnasium as gym
import jax
import jax.numpy as jnp

import reverb
import flax.linen as nn

from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass

import structlog
from flax import struct
from flax.training import train_state
from gymnasium import spaces
from jax import Array, jit

import distrax

LOG = structlog.getLogger()

ParamsType = Dict[str, Any]


@flax.struct.dataclass
class ExpConfig:
  max_replay_size: int = int(1e6)
  min_replay_size: int = 256  # int(1e3)
  num_rw_workers: int = 5
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
  num_eval_iterations: int = 1
  eval_frequency: int = 100
  steps_per_model_clock: int = 300


def rng_seq(*, seed: Optional[int | ArrayLike] = None, rng_key: Optional[Array] = None):
  assert seed is not None or rng_key is not None

  if rng_key is None:
    assert seed is not None
    rng_key = jax.random.PRNGKey(seed)

  while True:
    rng_key, sub_key = jax.random.split(rng_key)
    yield sub_key


class MetricWriter(Protocol):

  def scalar(self, tag: str, value: float, step: int):
    pass


@dataclasses.dataclass(frozen=True)
class RWConfig:
  reverb_address: str
  reverb_table_name: str


def rw_test(shutdown: EventClass):
  while not shutdown.is_set():
    print("go")
    time.sleep(2)
  print('terminating')


def as_float32(data: Any) -> np.ndarray:
  return np.asarray(data, dtype=np.float32)


class TrainState(train_state.TrainState):
  pass


class QTrainState(train_state.TrainState):
  target_params: FrozenDict[str, Any] = struct.field(pytree_node=True)


class SACModelState(struct.PyTreeNode):
  policy_state: TrainState
  q1_state: QTrainState
  q2_state: QTrainState

  # I think the cleanest way to wrap up alpha would be to put it in a TrainState as well, but I'm
  # making the choice not to so that I have practice manual applying the transformations and tracking state
  alpha_params: VariableDict
  alpha_optimizer_params: optax.GradientTransformation

  model_clock: jax.Array


@dataclasses.dataclass()
class Transition:
  obs: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  terminated: np.ndarray
  truncated: np.ndarray
  info: Dict[str, Any]


def collect(env: gym.Env, policy: nn.Module, params: VariableDict, rng_key: Array, exploit=False) -> Tuple[
  float, List[Transition]]:
  rng_gen = rng_seq(rng_key=rng_key)
  the_return = 0.
  obs, _ = env.reset(seed=next(rng_gen)[0].item())
  transitions = []
  while True:
    # TODO: For the policy I briefly got distracted starting to abstract this, DON'T DO IT!!! (yet).
    action, log_p, exploit_action = policy.apply({"params": params}, jnp.asarray(obs), next(rng_gen))
    action = exploit_action if exploit else action
    action = jnp.clip(action[0], a_min=-1, a_max=1)
    # TODO: move scaling to the appropriate place
    next_obs, reward, terminated, truncated, info = env.step(np.asarray(action * 2))
    reward = float(reward)
    transitions.append(
      Transition(obs=obs,
                 action=action,
                 reward=as_float32(reward),
                 terminated=as_float32(terminated),
                 truncated=as_float32(truncated),
                 info=info))
    the_return += reward
    obs = next_obs

    if terminated or truncated:
      transitions.append(
        Transition(obs=obs,
                   action=action,
                   reward=as_float32(reward),
                   terminated=as_float32(terminated),
                   truncated=as_float32(truncated),
                   info=info))
      break

  return the_return, transitions


def rw(idx: int, shutdown: EventClass, queue, config: RWConfig):
  # try:
  with jax.default_device(jax.devices('cpu')[0]):
    assert jnp.array([0]).devices().pop().platform == "cpu"
    rw_(idx, shutdown, queue, config)
  # except Exception as e:
  #   raise e


def rw_(rw_id: int, shutdown: EventClass, queue: Queue, config: RWConfig):
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

    with reverb_client.trajectory_writer(num_keep_alive_refs=2) as writer:
      for idx, step in enumerate(trajectory):
        writer.append(dataclasses.asdict(step))
        if idx > 0:
          try:
            writer.create_item(table=config.reverb_table_name,
                               trajectory={
                                 "obs": writer.history["obs"][idx - 1],
                                 "action": writer.history["action"][idx - 1],
                                 "reward": writer.history["reward"][idx - 1],
                                 "terminated": writer.history["terminated"][idx - 1],
                                 "next_obs": writer.history["obs"][idx],
                               }, priority=1)
          except Exception as e:
            raise e
        writer.flush()

  LOG.info(f"{rw_id} shutting down rw")


def eval_process(shutdown: EventClass, model_queue: multiprocessing.Queue, metric_queue: multiprocessing.Queue,
                 config: ExpConfig, rng_key):
  with jax.default_device(jax.devices('cpu')[0]):
    rng_gen = rng_seq(rng_key=rng_key)
    metric_writer = QueueMetricWriter(metric_queue)

    while not shutdown.is_set():
      try:
        data = model_queue.get(timeout=1)
        model_data = flax.serialization.msgpack_restore(bytearray(data))
        params = model_data["policy_params"]
        model_clock = model_data["model_clock"]
        if model_clock % config.eval_frequency == 0:
          LOG.info(f"Starting eval for model_clock: {model_clock}")
          eval_step(params, model_clock, config, next(rng_gen), metric_writer)

      except Empty:
        pass


def eval_step(policy_params: VariableDict, model_clock: int, config: ExpConfig, rng_key: Array,
              metric_writer: MetricWriter):
  env = env_factory()
  policy = policy_factory(env)

  returns = []
  for i in range(config.num_eval_iterations):
    the_return, _ = collect(env, policy, policy_params, rng_key=rng_key, exploit=True)
    returns.append(the_return)

  metric_writer.scalar("eval_return", np.mean(returns), model_clock)


class Policy(nn.Module):
  action_size: int
  scale: int

  @nn.compact
  def __call__(self, observations: Array, rng_key: Array):
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
    # dist = norm

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


@jit
def train_step(batch: Dict[str, Array], model_state: SACModelState, config: ExpConfig, rng_key: Array) -> Tuple[
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

  observations: Array = atleast_2d(batch["obs"])
  actions: Array = atleast_2d(batch["action"])
  rewards: Array = atleast_2d(batch["reward"])
  dones: Array = atleast_2d(batch["terminated"])
  next_observations: Array = atleast_2d(batch["next_obs"])
  metrics = {}

  policy_state = model_state.policy_state
  q1_state = model_state.q1_state
  q2_state = model_state.q2_state
  alpha = model_state.alpha_params["alpha"][0]

  # ---- Q-function updates
  next_sampled_actions, next_sampled_actions_logits, *_ = policy_state.apply_fn(
    {"params": policy_state.params}, next_observations, next(rng_gen))

  target_values_1: Array = q1_state.apply_fn({"params": q1_state.target_params},
                                             next_observations, next_sampled_actions)
  target_values_2: Array = q2_state.apply_fn({"params": q2_state.target_params},
                                             next_observations, next_sampled_actions)

  q_target = rewards + config.gamma * (1 - dones) * (jnp.minimum(target_values_1,
                                                                 target_values_2) -
                                                     alpha * next_sampled_actions_logits)

  # Note to self: by default, value_and_grad will take the derivate of the loss (first returned val) wrt the first
  # param, so the params that we want grads for need to be the first argument.
  def q_loss_fn(q_state_params: VariableDict, q_state: TrainState, q_target: Array, states: Array, actions: Array):
    predicted_q = q_state.apply_fn({"params": q_state_params}, states, actions)
    return jnp.mean(jnp.square(predicted_q - q_target))

  q_grad_fn = jax.value_and_grad(q_loss_fn, has_aux=False)
  q1_loss, grads = q_grad_fn(q1_state.params, q1_state, q_target, observations, actions)
  q1_state = q1_state.apply_gradients(grads=grads)
  q2_loss, grads = q_grad_fn(q2_state.params, q2_state, q_target, observations, actions)
  q2_state = q2_state.apply_gradients(grads=grads)
  metrics["loss.q"] = (q1_loss + q2_loss) / 2.

  def update_target_network(q_state: QTrainState) -> QTrainState:
    target_params = jax.tree_map(lambda source, target: (1 - config.tau) * source + config.tau * target, q_state.params,
                                 q_state.target_params)
    q_state = q_state.replace(target_params=target_params)

    return q_state

  q1_state = update_target_network(q1_state)
  q2_state = update_target_network(q2_state)

  # ------ Policy updates

  def policy_loss_fn(policy_params: VariableDict, observations: Array, rng_key: Array):
    actions, logits, *_ = policy_state.apply_fn({"params": policy_params}, observations, rng_key)
    q_1 = q1_state.apply_fn({"params": q1_state.params}, observations, actions)
    q_2 = q2_state.apply_fn({"params": q2_state.params}, observations, actions)

    min_q = jnp.minimum(q_1, q_2)
    loss = jnp.mean(alpha * logits - min_q)

    return loss

  policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
  policy_loss, grads = policy_grad_fn(policy_state.params, observations, next(rng_gen))
  policy_state = policy_state.apply_gradients(grads=grads)
  metrics["loss.policy"] = policy_loss

  # ------- Alpha update

  actions, log_p_actions, *_ = policy_state.apply_fn({"params": policy_state.params}, observations, next(rng_gen))

  # heuristic used in the original paper and codebase
  target_entropy = -jnp.prod(jnp.array(actions.shape[1:]))

  alpha_loss, grads = jax.value_and_grad(
    lambda alpha_params: -(alpha_params["alpha"] * (log_p_actions + target_entropy)).mean(), has_aux=False)(
    model_state.alpha_params)
  updates, alpha_optimizer_params = optax.adam(learning_rate=config.alpha_lr).update(grads,
                                                                                     model_state.alpha_optimizer_params,
                                                                                     model_state.alpha_params)
  alpha_params = optax.apply_updates(model_state.alpha_params, updates)
  metrics["loss.alpha"] = alpha_loss

  metrics["alpha"] = alpha_params["alpha"][0]

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


class QueueMetricWriter(MetricWriter):

  def __init__(self, queue: Queue):
    self.queue = queue

  def scalar(self, tag: str, value: float, step: int):
    self.queue.put_nowait({"tag": tag, "value": value, "step": step})


@dataclasses.dataclass(frozen=True)
class Models:
  policy: nn.Module
  q_1: nn.Module
  q_2: nn.Module
  q_1_target: nn.Module
  q_2_target: nn.Module


def create_policy(env: gym.Env, config: ExpConfig, rng_key: Array) -> Tuple[nn.Module, TrainState]:
  rng_gen = rng_seq(rng_key=rng_key)
  policy = Policy(action_size=env.action_space.shape[0], scale=2)
  init_samples = [env.observation_space.sample(), env.observation_space.sample()]
  output, policy_variables = policy.init_with_output(next(rng_gen), jnp.array(init_samples), next(rng_gen))
  policy_state = TrainState.create(
    apply_fn=policy.apply,
    params=policy_variables["params"],
    tx=optax.adam(learning_rate=config.policy_learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2)
  )

  return policy, policy_state


def create_q_function(env: gym.Env, config: ExpConfig, rng_key: Array) -> Tuple[nn.Module, QTrainState]:
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

  return q, q_state


@dataclasses.dataclass(frozen=True)
class RWModel:
  model_clock: int
  params: Dict[str, Any]
  model_factory: Callable[[], nn.Module]


def env_factory() -> gym.Env:
  return gym.make("Pendulum-v1")


def policy_factory(env: gym.Env) -> Policy:
  action_size = env.action_space.shape[0]
  # TODO: hardcoding this is bad
  return Policy(action_size=action_size, scale=2)


def atleast_2d(data: Array) -> Array:
  if len(data.shape) < 2:
    data = jnp.expand_dims(data, -1)

  return data


def convert_batch(batch: ReplaySample) -> Dict[str, jnp.ndarray]:
  return {k: atleast_2d(jnp.asarray(v)) for k, v in batch.data.items()}


def main():
  """
  Note: moving everything into a main function, instead of leaving it tucked under if __name__=="__main__", can
  address two problems:
    - It can be written in such a way as to be a reusable entry point, callable from different scripts (not done here).
    - It prevents variables from being exposed as global. Global vars caused at least one problem for me here when
    using autocomplete in my IDE.
  """

  multiprocessing.set_start_method("spawn")
  config = ExpConfig()

  assert jnp.array([0]).devices().pop().platform == "gpu"

  seed = config.seed if config.seed is not None else time.time_ns()
  rng_gen = rng_seq(seed=seed)

  summary_writer = SummaryWriter(
    f"data/classic_control/pendulum/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}")

  env = env_factory()

  assert type(env.action_space) == spaces.Box
  assert type(env.observation_space) == spaces.Box

  policy, policy_state = create_policy(env, config, next(rng_gen))
  q1, q1_state, = create_q_function(env, config, next(rng_gen))
  q2, q2_state, = create_q_function(env, config, next(rng_gen))

  # although alpha is a scalar, it needs to have a dimension for jax.grad to be happy
  alpha_params = {"alpha": jnp.array([config.alpha])}

  sac_state = SACModelState(policy_state=policy_state,
                            q1_state=q1_state,
                            q2_state=q2_state,
                            alpha_params=alpha_params,
                            alpha_optimizer_params=optax.adam(learning_rate=config.alpha_lr).init(alpha_params),
                            model_clock=jnp.array(0, dtype=jnp.int32), )

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
        "obs": tf.TensorSpec([*env.observation_space.shape], env.observation_space.dtype),
        "action": tf.TensorSpec([*env.action_space.shape], env.action_space.dtype),
        "reward": tf.TensorSpec((), tf.float32),
        "terminated": tf.TensorSpec((), tf.float32),
        "next_obs": tf.TensorSpec([*env.observation_space.shape], env.observation_space.dtype),
      }
    )
  ])

  reverb_address = f'localhost:{replay_server.port}'

  replay_client = reverb.Client(reverb_address)

  dataset = reverb.TrajectoryDataset.from_table_signature(server_address=reverb_address, table=reverb_table_name,
                                                          max_in_flight_samples_per_worker=config.batch_size * 2).batch(
    config.batch_size)

  rw_config = RWConfig(reverb_address=reverb_address, reverb_table_name=reverb_table_name)

  # ------ Set up processes for data collection.
  terminate_event: EventClass = Event()
  lock = multiprocessing.Lock()

  model_queues = [multiprocessing.Queue() for i in range(config.num_rw_workers)]
  processes = [Process(target=rw, args=(i, terminate_event, model_queues[i], rw_config)) for i in
               range(config.num_rw_workers)]

  metric_queue = multiprocessing.Queue()
  eval_model_queue = multiprocessing.Queue()
  model_queues.append(eval_model_queue)

  processes.append(
    Process(target=eval_process, args=(terminate_event, eval_model_queue, metric_queue, config, next(rng_gen))))
  for p in processes:
    p.start()

  def send_model(model_clock: int) -> None:
    LOG.info(f"Sending model: {model_clock}")
    msg = flax.serialization.msgpack_serialize(
      {"policy_params": sac_state.policy_state.params, "model_clock": model_clock})
    for q in model_queues:
      q.put(msg)

  send_model(model_clock=0)

  # training loop
  while True:
    # LOG.info(f"Replay buffer size: {replay_client.server_info()[reverb_table_name].current_size}")
    if replay_client.server_info()[reverb_table_name].current_size < config.min_replay_size:
      time.sleep(1)
      continue

    batch = convert_batch(list(dataset.take(1))[0])

    new_sac_state, metrics = train_step(batch, sac_state, config=config, rng_key=next(rng_gen))

    sac_state = new_sac_state
    model_clock = int(sac_state.model_clock)

    send_model(model_clock)

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


# TODO: model save and load
# TODO: I noticed something with the eval runs producing exactly the same returns, investigate.
# TODO: adaptive alpha
# TODO: seeing GPU out of memory errors

if __name__ == "__main__":
  main()
