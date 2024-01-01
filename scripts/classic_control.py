"""
This is a training script to first train gym agents in the classic control setting to
validate my SAC implementation.
"""
import ctypes
import multiprocessing
import os
from queue import Empty

import flax.serialization
import numpy as np
import optax
from flax.core import FrozenDict
from flax.core.scope import VariableDict
from jax._src.basearray import ArrayLike
from reverb import ReplaySample

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu"

import dataclasses
import time
from typing import Dict, Any, Optional, Tuple, List, Callable, Protocol, Mapping

import gymnasium as gym
import jax
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import reverb
import flax.linen as nn

from multiprocessing import Pool, Event, Value, Process
from multiprocessing.synchronize import Event as EventClass
from multiprocessing import Value as ValueClass

import structlog
from flax import struct
from flax.training import train_state
from gymnasium import spaces
from jax import Array, jit
import jax.numpy as jnp
import distrax

LOG = structlog.getLogger()

ParamsType = Dict[str, Any]

@flax.struct.dataclass
class ExpConfig:
  max_replay_size: int = int(1e6)
  min_replay_size: int = int(1e3)
  num_rw_workers: int = 5
  seed: Optional[int] = None
  learning_rate: float = 0.001
  adam_beta_1: float = 0.5
  adam_beta_2: float = 0.999
  batch_size: int = 256
  gamma: float = 0.99
  tau: float = 0.005 # soft-target update param


def rng_seq(*, seed: Optional[int | ArrayLike] = None, rng_key: Optional[Array] = None):
  assert seed is not None or rng_key is not None

  if rng_key is None:
    assert seed is not None
    rng_key = jax.random.PRNGKey(seed)

  while True:
    rng_key, sub_key = jax.random.split(rng_key)
    yield sub_key


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


def rw(idx: int, shutdown: EventClass, queue, config: RWConfig):
  try:
    with jax.default_device(jax.devices('cpu')[0]):
      rw_(idx, shutdown, queue, config)
  except Exception as e:
    raise e


def rw_(idx: int, shutdown: EventClass, queue, config: RWConfig):
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
          received = queue.get_nowait()
      except Empty:
        pass

      result = flax.serialization.msgpack_restore(bytearray(received)) if received is not None else current

      if result is None:
        time.sleep(0.1)

    return result

  @dataclasses.dataclass()
  class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: Dict[str, Any]

  LOG.info(f"rw{idx} process starting")
  # os.environ["JAX_PLATFORMS"] = "cpu"
  # with jax.default_device(jax.devices('cpu')[0]):

  reverb_client = reverb.Client(config.reverb_address)
  env = env_factory()

  policy = policy_factory(env)

  def collect(params: VariableDict, rng_key: Array) -> Tuple[float, List[Transition]]:
    LOG.info("starting episode")
    rng_gen = rng_seq(rng_key=rng_key)
    the_return = 0.
    obs, _ = env.reset(seed=next(rng_gen)[0].item())
    transitions = []
    while True:
      # TODO: For the policy I briefly got distracted starting to abstract this, DON'T DO IT!!! (yet).
      action, *_ = policy.apply({"params": params}, jnp.asarray(obs), next(rng_gen))
      next_obs, reward, terminated, truncated, info = env.step(np.asarray(action))
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

  rng_gen = rng_seq(seed=time.time_ns())

  rw_model = None
  while not shutdown.is_set():
    print(f"{idx} running")

    rw_model = get_latest(rw_model)

    model_clock = rw_model["model_clock"]
    policy_params = rw_model["policy_params"]
    LOG.info(f"Received model {model_clock}")

    the_return, trajectory = collect(policy_params, next(rng_gen))

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

  LOG.info("shutting down rw")


class Policy(nn.Module):
  action_size: int

  @nn.compact
  def __call__(self, observations: Array, rng_key: Array):
    rng_gen = rng_seq(rng_key=rng_key)

    # TODO: for the moment I'm hardcoding some vals just for pendulum
    x = nn.Sequential([
      nn.Dense(16),
      nn.relu,
      nn.Dense(16),
      nn.relu,
    ])(observations)

    means = nn.Dense(self.action_size)(x)
    # log_std_dev is defined on [-inf, inf]
    log_std_dev = nn.Dense(self.action_size)(x)
    std_dev = jnp.exp(log_std_dev)

    norm = distrax.MultivariateNormalDiag(loc=means, scale_diag=std_dev)
    dist = distrax.Transformed(distribution=norm, bijector=distrax.Block(distrax.Tanh(), ndims=self.action_size))

    actions, action_log_prob = dist.sample_and_log_prob(seed=next(rng_gen))

    return actions, jnp.expand_dims(action_log_prob, -1), means, log_std_dev


# class FlaxPolicy(PolicyProtocol):
#
#   def __init__(self):
#
#
#   def __call__(self, obs: np.ndarray, params: Any) -> np.ndarray:
#     j_obs = jnp.array(obs)


class QFunction(nn.Module):

  @nn.compact
  def __call__(self, states: Array, actions: Array) -> Array:
    inputs = jnp.concatenate([states, actions], axis=1)

    return nn.Sequential([
      nn.Dense(16),
      nn.relu,
      nn.Dense(16),
      nn.relu,
      nn.Dense(1)
    ])(inputs)


class TrainState(train_state.TrainState):
  pass


class QTrainState(train_state.TrainState):
  target_params: FrozenDict[str, Any] = struct.field(pytree_node=True)


class SACModelState(struct.PyTreeNode):
  policy_state: TrainState
  q1_state: QTrainState
  q2_state: QTrainState
  model_clock: int = 0


@jit
def train_step(batch: Dict[str, Array], model_state: SACModelState, config: ExpConfig, rng_key: Array) -> Tuple[
  SACModelState, Dict[str, float]]:
  rng_gen = rng_seq(rng_key=rng_key)

  observations: Array = atleast_2d(batch["obs"])
  actions: Array = atleast_2d(batch["action"])
  rewards: Array = atleast_2d(batch["reward"])
  dones: Array = atleast_2d(batch["terminated"])
  next_observations: Array = atleast_2d(batch["next_obs"])
  losses = {}

  policy_state = model_state.policy_state
  q1_state = model_state.q1_state
  q2_state = model_state.q2_state

  next_sampled_actions, next_sampled_actions_logits, *_ = policy_state.apply_fn(
    {"params": model_state.policy_state.params}, next_observations, next(rng_gen))

  target_values_1: Array = q1_state.apply_fn({"params": q1_state.target_params},
                                             next_observations, next_sampled_actions)
  target_values_2: Array = q2_state.apply_fn({"params": q2_state.target_params},
                                             next_observations, next_sampled_actions)

  # TODO: adaptive
  alpha: float = 0.5

  # TODO: what value of rho to use?
  rho: float = 0.5

  q_target = rewards + config.gamma * (1 - dones) * jnp.minimum(target_values_1,
                                                         target_values_2) - alpha * next_sampled_actions_logits

  # Note to self: by default, we take the derivate of the first param wrt to the inputs. So params should be first.
  def q_loss_fn(q_state_params: VariableDict, q_state: TrainState, q_target: Array, states: Array, actions: Array):
    predicted_q = q_state.apply_fn({"params": q_state_params}, states, actions)
    return jnp.mean(jnp.square(predicted_q - q_target))

  q_grad_fn = jax.value_and_grad(q_loss_fn, has_aux=False)
  q1_loss, grads = q_grad_fn(q1_state.params, q1_state, q_target, observations, actions)
  q1_state = q1_state.apply_gradients(grads=grads)
  q2_loss, grads = q_grad_fn(q2_state.params, q2_state, q_target, observations, actions)
  q2_state = q2_state.apply_gradients(grads=grads)
  losses["q"] = (q1_loss + q2_loss) / 2.

  def policy_loss_fn(policy_params: VariableDict, states: Array, rng_key: Array):
    actions, logits, *_ = policy_state.apply_fn({"params": policy_params}, states, rng_key)
    q_1 = q1_state.apply_fn({"params": q1_state.params}, states, actions)
    q_2 = q2_state.apply_fn({"params": q2_state.params}, states, actions)

    min_q = jnp.minimum(q_1, q_2)
    loss = jnp.mean(min_q - alpha * logits)

    return loss

  policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=False)
  policy_loss, grads = policy_grad_fn(policy_state.params, observations, next(rng_gen))
  policy_state = model_state.policy_state.apply_gradients(grads=grads)

  losses["policy"] = policy_loss

  def update_target_network(q_state: QTrainState) -> QTrainState:
    target_params = jax.tree_map(lambda source, target: (1 - config.tau) * source + config.tau * target, q_state.params,
                                 q_state.target_params)
    q_state = q_state.replace(target_params=target_params)

    return q_state

  q1_state = update_target_network(q1_state)
  q2_state = update_target_network(q2_state)

  return SACModelState(
    model_clock=sac_state.model_clock + 1,
    policy_state=policy_state,
    q1_state=q1_state,
    q2_state=q2_state,
  ), losses


@dataclasses.dataclass(frozen=True)
class Models:
  policy: nn.Module
  q_1: nn.Module
  q_2: nn.Module
  q_1_target: nn.Module
  q_2_target: nn.Module


def create_policy(env: gym.Env, config: ExpConfig) -> Tuple[nn.Module, TrainState]:
  policy = Policy(action_size=env.action_space.shape[0])
  init_samples = [env.observation_space.sample(), env.observation_space.sample()]
  output, policy_variables = policy.init_with_output(next(rng_gen), jnp.array(init_samples), next(rng_gen))
  policy_state = TrainState.create(
    apply_fn=policy.apply,
    params=policy_variables["params"],
    tx=optax.adam(learning_rate=config.learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2)
  )

  return policy, policy_state


def create_q_function(env: gym.Env, config: ExpConfig) -> Tuple[nn.Module, QTrainState]:
  q = QFunction()
  init_samples = jnp.array([env.observation_space.sample(), env.observation_space.sample()])
  init_actions = jnp.array([env.action_space.sample(), env.action_space.sample()])
  output, q_variables = q.init_with_output(next(rng_gen), states=jnp.array(init_samples), actions=init_actions)
  q_state = QTrainState.create(
    apply_fn=q.apply,
    params=q_variables["params"],
    tx=optax.adam(learning_rate=config.learning_rate, b1=config.adam_beta_1, b2=config.adam_beta_2),
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
  return Policy(action_size=action_size)


def atleast_2d(data: Array) -> Array:
  if len(data.shape) < 2:
    data = jnp.expand_dims(data, -1)

  return data


def convert_batch(batch: ReplaySample) -> Dict[str, jnp.ndarray]:
  return {k: atleast_2d(jnp.asarray(v)) for k, v in batch.data.items()}


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn")
  config = ExpConfig()

  seed = config.seed if config.seed is not None else time.time_ns()
  rng_gen = rng_seq(seed=seed)

  env = env_factory()

  assert type(env.action_space) == spaces.Box
  assert type(env.observation_space) == spaces.Box

  policy, policy_state = create_policy(env, config)
  q1, q1_state, = create_q_function(env, config)
  q2, q2_state, = create_q_function(env, config)

  sac_state = SACModelState(policy_state=policy_state, q1_state=q1_state, q2_state=q2_state)

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

  rw_config = RWConfig(reverb_address=reverb_address, reverb_table_name=reverb_table_name)

  dataset = reverb.TrajectoryDataset.from_table_signature(server_address=reverb_address, table=reverb_table_name,
                                                          max_in_flight_samples_per_worker=config.batch_size * 2).batch(
    config.batch_size)

  terminate_event: EventClass = Event()
  lock = multiprocessing.Lock()

  queues = [multiprocessing.Queue() for i in range(config.num_rw_workers)]
  processes = [Process(target=rw, args=(i, terminate_event, queues[i], rw_config)) for i in
               range(config.num_rw_workers)]
  for p in processes:
    p.start()


  def send_model(model_clock):
    msg = flax.serialization.msgpack_serialize(
      {"policy_params": sac_state.policy_state.params, "model_clock": model_clock})
    for q in queues:
      q.put_nowait(msg)


  # try:

  send_model(model_clock=0)

  # training loop
  while True:
    LOG.info(f"Replay buffer size: {replay_client.server_info()[reverb_table_name].current_size}")
    if replay_client.server_info()[reverb_table_name].current_size < config.min_replay_size:
      time.sleep(30)
      continue

    batch = convert_batch(list(dataset.take(1))[0])
    sac_state, metrics = train_step(batch, sac_state, config=config, rng_key=next(rng_gen))

    send_model(sac_state.model_clock)
    print({k: float(v) for k, v in metrics.items()})

  # except Exception as e:
  #   print(e)

  terminate_event.set()

  for p in processes:
    p.join()

  replay_server.stop()

  print("exit")
