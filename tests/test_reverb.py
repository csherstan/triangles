import pytest

from triangles.sac import (
    write_trajectory,
    TransitionStep,
)
from triangles.util import as_float32, space_to_reverb_spec
import reverb
import gymnasium as gym
import tensorflow as tf
import numpy as np

# do not remove, needed for MixedAction2D to be registered with gym.
import triangles.env.mixed_action # noqa


@pytest.mark.parametrize("env_name", ["Pendulum-v1", "MixedAction2D-v0"])
def test_write_trajectory(env_name: str):
    """
    Writes a short trajectory to the reverb buffer, reads it back out and checks that everything is in
    the correct order.

    The test is conducted for Pendulum, which has a Box action and observation space, and MixedAction2D, which has
    a Dict action space. This required extra work in order to accommodate both.

    :param env_name: name of the gym env to use for testing
    :return:
    """

    env = gym.make(env_name)

    trajectory = [
        TransitionStep(
            obs=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=as_float32(1.0),
            terminated=as_float32(False),
            truncated=as_float32(False),
            info={},
        ),
        TransitionStep(
            obs=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=as_float32(2.0),
            terminated=as_float32(False),
            truncated=as_float32(False),
            info={},
        ),
        TransitionStep(
            obs=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=as_float32(3.0),
            terminated=as_float32(False),
            truncated=as_float32(False),
            info={},
        ),
    ]

    reverb_table_name = "table"
    # create reverb server
    replay_server = reverb.Server(
        tables=[
            reverb.Table(
                name=reverb_table_name,
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_times_sampled=1,
                max_size=len(trajectory) - 1,
                rate_limiter=reverb.rate_limiters.MinSize(1),
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

    reverb_client = reverb.Client(reverb_address)

    write_trajectory(reverb_client, reverb_table_name, trajectory)

    assert (
        reverb_client.server_info()[reverb_table_name].current_size
        == len(trajectory) - 1
    )

    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=reverb_address,
        table=reverb_table_name,
        max_in_flight_samples_per_worker=1,
        max_samples=1,
    )
    sample = list(dataset.take(1))[0]

    print(sample)

    assert np.all(sample.data["obs"] == trajectory[0].obs)
    assert np.all(sample.data["action"] == trajectory[0].action)
    assert np.all(sample.data["reward"] == trajectory[0].reward)
    assert np.all(sample.data["terminated"] == trajectory[0].terminated)
    assert np.all(sample.data["next_obs"] == trajectory[1].obs)

    sample = list(dataset.take(1))[0]
    assert np.all(sample.data["obs"] == trajectory[1].obs)
    assert np.all(sample.data["next_obs"] == trajectory[2].obs)
