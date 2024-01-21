# Copyright Craig Sherstan 2024

"""
A utility script for measuring processing time for several processes:
1. The rate at which MountainCar runs
2. The throughput of the policy when run on GPU
3. The throughput of the policy when run on CPU
4. The throughput of the collect operation when run on CPU

It appears that running the policy forward on CPU is faster than GPU.
Further, running the policy forward is the bottleneck, not the environment itself.
"""

import time

import gymnasium as gym
import jax
import jax.numpy as jnp

from triangles.model.continuous import create_policy_state, policy_factory, PolicyWrapper
from triangles.sac import ExpConfig, collect
from triangles.util import rng_seq

"""
Env timing: 198971.9117898456 steps/s, 5.025835008592577e-06 s/step
CPU
{CpuDevice(id=0)}
CPU apply 82.67215937948764 steps/s, 0.012095970487594605 s/step
GPU
{cuda(id=0)}
GPU apply 56.51047245180891 steps/s, 0.017695835065841676 s/step
collect CPU
collect 81.27861678270006 steps/s, 0.012303358983993531 s/step
"""

# one episode max is 1000 steps

num_episodes = 5
num_steps = num_episodes * 1000

env = gym.make("MountainCarContinuous-v0")
config = ExpConfig()

# measure the throughput on MountainCar by running multiple episodes and recording the time it takes and the number
# of steps taken
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

# measure the throughput of the Policy when using CPU
print("CPU")
with jax.default_device(jax.devices("cpu")[0]):
    # need to recreate all the variables on the correct device
    rng_gen = rng_seq(seed=0)
    policy = policy_factory(env)
    assert isinstance(policy, PolicyWrapper)
    policy_state = create_policy_state(
        env=env, policy=policy, config=config, rng_key=next(rng_gen)
    )
    obs = jnp.array(env.observation_space.sample())
    print(obs.devices())

    # Call the policy repeatedly and measure the total time taken
    start_time = time.time()
    for i in range(num_steps):
        policy({"params": policy_state.params}, obs, next(rng_gen))
    delta_time = time.time() - start_time
    print(
        f"CPU apply {num_steps / delta_time} steps/s, {delta_time / num_steps} s/step"
    )

# measure the throughput of the Policy when using GPU
print("GPU")
with jax.default_device(jax.devices("gpu")[0]):

    # need to recreate all the variables on the correct device
    rng_gen = rng_seq(seed=0)
    policy = policy_factory(env)
    assert isinstance(policy, PolicyWrapper)
    policy_state = create_policy_state(
        env=env, policy=policy, config=config, rng_key=next(rng_gen)
    )
    obs = jnp.array(env.observation_space.sample())
    print(obs.devices())

    # Call the policy repeatedly and measure the total time taken
    start_time = time.time()
    for i in range(num_steps):
        policy({"params": policy_state.params}, obs, next(rng_gen))
    delta_time = time.time() - start_time
    print(
        f"GPU apply {num_steps / delta_time} steps/s, {delta_time / num_steps} s/step"
    )

# Measure the throughput of the collect step on CPU
print("collect CPU")
with jax.default_device(jax.devices("cpu")[0]):
    rng_gen = rng_seq(seed=0)
    policy_state = create_policy_state(
        env=env, policy=policy, config=config, rng_key=next(rng_gen)
    )

    # now we're going to collect multiple complete trajectories and compute the throughput.
    # unlike previous steps, this requires calling the policy, getting new observations, etc.
    start_time = time.time()
    transition_count = 0
    for i in range(num_episodes):
        _, transitions = collect(
            env, policy, policy_params=policy_state.params, rng_key=next(rng_gen)
        )
        transition_count += len(transitions)
    delta_time = time.time() - start_time
    print(
        f"collect {transition_count / delta_time} steps/s, {delta_time / transition_count} s/step"
    )
