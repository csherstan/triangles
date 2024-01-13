import pytest

from triangles.common import write_trajectory, space_to_reverb_spec, TransitionStep, as_float32
import reverb
import gymnasium as gym
import tensorflow as tf
import triangles.env.mixed_action
import numpy as np

@pytest.mark.parametrize("env_name", ["Pendulum-v1", "MixedAction2D-v0"])
def test_write_trajectory(env_name: str):
    env = gym.make(env_name)

    reverb_table_name = 'table'
    # create reverb server
    replay_server = reverb.Server(tables=[
        reverb.Table(
            name=reverb_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=1,
            rate_limiter=reverb.rate_limiters.MinSize(1),
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

    reverb_client = reverb.Client(reverb_address)

    trajectory = [
        TransitionStep(
            obs=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=as_float32(1.0),
            terminated=as_float32(False),
            truncated=as_float32(False),
            info={}
        ),
        TransitionStep(
            obs=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=as_float32(1.0),
            terminated=as_float32(False),
            truncated=as_float32(False),
            info={}
        )
    ]

    write_trajectory(reverb_client, reverb_table_name, trajectory)

    assert reverb_client.server_info()[reverb_table_name].current_size == 1

    dataset = reverb.TrajectoryDataset.from_table_signature(server_address=reverb_address,
                                                            table=reverb_table_name,
                                                            max_in_flight_samples_per_worker=2).batch(1)
    sample = list(dataset.take(1))[0]

    print(sample)

    assert np.all(sample.data["obs"] == trajectory[0].obs)
    assert np.all(sample.data["action"] == trajectory[0].action)
    assert np.all(sample.data["reward"] == trajectory[0].reward)
    assert np.all(sample.data["terminated"] == trajectory[0].terminated)
    assert np.all(sample.data["next_obs"] == trajectory[1].obs)