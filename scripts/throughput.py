import time

import gymnasium as gym
import jax
import jax.numpy as jnp

from triangles.model.continuous import create_policy_state, policy_factory
from triangles.common import rng_seq, ExpConfig, collect

"""
Env timing: 201221.20767982173 steps/s, 4.969655095158635e-06 s/step
CPU
{CpuDevice(id=0)}
CPU apply 57.569575565068185 steps/s, 0.01737028613090515 s/step
GPU
{cuda(id=0)}
GPU apply 49.06050859067015 steps/s, 0.020382992935180665 s/step
collect CPU
collect 50.33346624613468 steps/s, 0.019867497205734254 s/step
"""

# one episode max is 1000 steps

num_episodes = 5
num_steps = num_episodes * 1000

env = gym.make("MountainCarContinuous-v0")
policy = policy_factory(env)
config = ExpConfig()
rng_gen = rng_seq(seed=0)
policy_state = create_policy_state(env=env, policy=policy, config=config, rng_key=next(rng_gen))

action = env.action_space.sample()
start_time = time.time()
count = 0
for i in range(num_episodes):
  env.reset()
  while True:
    _, _, truncated, terminated, _ = env.step(action)
    count += 1
    if truncated or terminated:
      break
delta_time = time.time() - start_time
print(f"Env timing: {count / delta_time} steps/s, {delta_time / count} s/step")

print("CPU")
with jax.default_device(jax.devices('cpu')[0]):
  obs = jnp.array(env.observation_space.sample())
  print(obs.devices())

  start_time = time.time()
  for i in range(num_steps):
    policy.apply({"params": policy_state.params}, obs, next(rng_gen))
  delta_time = time.time() - start_time
  print(f"CPU apply {num_steps / delta_time} steps/s, {delta_time / num_steps} s/step")

print("GPU")
with jax.default_device(jax.devices('gpu')[0]):
  obs = jnp.array(env.observation_space.sample())
  print(obs.devices())

  start_time = time.time()
  for i in range(num_steps):
    policy.apply({"params": policy_state.params}, obs, next(rng_gen))
  delta_time = time.time() - start_time
  print(f"GPU apply {num_steps / delta_time} steps/s, {delta_time / num_steps} s/step")

print("collect CPU")
with jax.default_device(jax.devices('cpu')[0]):
  start_time = time.time()
  transition_count = 0
  for i in range(num_episodes):
    _, transitions = collect(env, policy, policy_params=policy_state.params, rng_key=next(rng_gen))
    transition_count += len(transitions)
  delta_time = time.time() - start_time
  print(f"collect {transition_count / delta_time} steps/s, {delta_time / transition_count} s/step")
