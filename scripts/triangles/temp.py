from PIL import Image

from triangles.common import collect
from triangles.model.triangles_nn import Policy
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import datasets

if __name__=="__main__":

    width = 300
    height = 600

    env = gym.make(
        "triangles-v1",
        width=width,
        height=height,
    )

    width = 300
    height = 600
    channels = 3
    image = jnp.zeros(shape=(height, width, channels))

    init_obs, _ = env.reset(options={"target": image})

    init_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), init_obs)

    policy = Policy()
    policy_output, policy_variables = policy.init_with_output(jax.random.PRNGKey(0), init_obs, jax.random.PRNGKey(0))
    the_return, transitions = collect(env, policy, policy_params=policy_variables["params"], rng_key=jax.random.PRNGKey(0))
    print(the_return, len(transitions))
    img: np.ndarray = env.render()
    Image.fromarray(img).show()