from functools import partial
from typing import List, Tuple

import distrax
from flax import nnx
import flax.linen as nn
import jax.random
from jax import Array
import jax.numpy as jnp

from triangles.common import DictArrayType
from triangles.env.triangle import TriangleEnv, TRIANGLE_SIZE


# from triangles.common import DictArrayType


# class CNN(nn.Module):
#
#     @nn.compact
#     def __call__(self, images: Array) -> Array:
#         x = nn.Sequential([
#             nn.Conv(features=64, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
#             nn.leaky_relu,
#             nn.Conv(features=128, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
#             nn.leaky_relu,
#             nn.Conv(features=256, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
#             nn.leaky_relu,
#             nn.Conv(features=512, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
#             nn.leaky_relu,
#             nn.Conv(features=1, kernel_sizes=(4, 4), strides=(1, 1), paddings="VALID", use_biases=False),
#             nn.leaky_relu,
#         ])(images)
#
#         return x

class CNN(nnx.Module):
    """A simple CNN model."""

    # this is just taken directly from the nnx examples

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(720000, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class MLP(nnx.Module):

    def __init__(self, in_features: int, features: List[int], activations: List[Array], *, rngs: nnx.Rngs):
        self.features = features
        self.activations = activations

        layers = []
        for feature, activation in zip(features, activations):
            layers.append(nnx.Linear(in_features=in_features, out_features=feature, rngs=rngs))
            if activation is not None:
                layers.append(activation)
            in_features = feature

        self.mlp = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.mlp(x)


class Encoder(nnx.Module):

    def __init__(self, in_features: int, embed, embed_idx, embed_size, rngs: nnx.Rngs):
        self.embed = embed
        self.embed_idx = embed_idx
        self.embed_size = embed_size
        self.encoder = MLP(in_features=in_features, features=[embed_size] * 2, activations=[nnx.selu, nnx.tanh],
                           rngs=rngs)

    def __call__(self, array: Array):
        assert len(array.shape) == 3    # not jit safe
        batch_size = array.shape[0]
        encoded = self.encoder(array) / jnp.sqrt(self.embed_size)
        embedding = self.embed(jnp.ones(shape=(batch_size), dtype=jnp.int32) * self.embed_idx)
        token = encoded + embedding
        return token


class Core(nnx.Module):

    def __init__(self, *, rngs: nnx.Rngs):
        self.num_heads: int = 4
        num_attention_layers: int = 4
        qkv_features: int = 24

        self.max_triangles = 20

        self.embedding_size = 16
        self.query_size = self.max_triangles + 1

        self.cnn = CNN(rngs=rngs)
        cnn_output_size = 10

        self.embed = nnx.Embed(num_embeddings=3, features=self.embedding_size, rngs=rngs)
        self.embedding_idx = {
            "error": 0,
            "triangles": 2,
            # "action": 1,
        }

        def make_encoder(in_features: int, idx: int):
            return Encoder(in_features=in_features, embed=self.embed, embed_idx=idx, embed_size=self.embedding_size,
                           rngs=rngs)

        self.error_encoder = make_encoder(cnn_output_size, self.embedding_idx["error"])
        # self.energy_encoder = MLP(1, [embedding_size] * 2, activations=[activation] * 2, rngs=rngs)

        # single triangle for the moment.
        # self.action_encoder = make_encoder(10, self.embedding_idx["action"])
        self.triangle_encoder = make_encoder(10, self.embedding_idx["triangles"])

        attention_blocks = []
        for i in range(num_attention_layers):
            attention_blocks.append(nnx.MultiHeadAttention(num_heads=self.num_heads, in_features=self.embedding_size,
                                                           qkv_features=qkv_features, decode=False, rngs=rngs))
        self.attention_blocks = nnx.Sequential(*attention_blocks)

        # self.q_network = MLP(in_features=embedding_size * (self.max_triangles + 2), features=[128, 128, 1],
        #                      activations=[activation] * 2 + [None], rngs=rngs)

    @property
    def out_features(self) -> int:
        return self.embedding_size*(self.query_size)

    @nn.compact
    def __call__(self, observations: DictArrayType) -> Array:
        """

        For edit/add ops: we don't add an input for the ops, instead we need to embed that information into the
        tokens that go into the transformer.

        Query system input:
        - CNN output embedding of image error
        - stop action

        I believe that one approach people take is to add additional info (like the stopping action) as an additional
        channel. Not sure what the right approach is here.

        I'm also thinking that maybe the CNN portion could be pretrained as a VAE. Or maybe I just grab a pretrained
        CNN

        :param observations:
        :param actions:
        :return:

        # encodings all need to be projected to the same size
        1. Encode the image error using a CNN
        2. Encode the available energy using an MLP
        3. Encode the selected action using an MLP
        4. Encode the triangles using an MLP

        2 and 3 above might be combined
        - Apply Embed to all the encodings.

        Feed into transformer stack
        - the output at this point is the set of vectors that went into the transformer
        - Apply an MLP with a single head for the Q-function.

        As a first pass:
        - Assume every action is an 'add' action
        - Fixed number of triangles, always output 20.

        """

        error_inputs = observations["error"]

        batch_size = error_inputs.shape[0]

        # expand dims to make second dim the token dim
        error_token = self.error_encoder(jnp.expand_dims(self.cnn(error_inputs), 1))

        # energy = None
        # action = actions["data"]["triangle"]
        #
        # action_tokens = self.action_encoder(action)

        triangles_buffer = jnp.zeros(shape=(batch_size, self.max_triangles, 10))
        triangles_obs = observations["triangles"]
        # TODO: I'm not sure how to properly handle this case yet. If I recall correctly, I think I need
        # to avoid if statements so I'm not sure how to handle checking for len 0 triangles
        triangles_buffer.at[:, 0:len(triangles_obs)].set(triangles_obs)

        triangle_tokens = self.triangle_encoder(triangles_buffer)

        tokens = jnp.concatenate([error_token, triangle_tokens], axis=1)


        mask = jnp.ones(shape=(batch_size, self.num_heads, self.query_size, self.query_size), dtype=jnp.bool)
        mask.at[:, :, :, -len(triangles_obs)].set(False)
        mask.at[:, :, -len(triangles_obs), :].set(False)

        outputs = self.attention_blocks(inputs_q=tokens, mask=mask)

        # flatten
        outputs = jnp.reshape(outputs, (-1, self.out_features))

        # q_value = self.q_network(outputs)

        return outputs


class QFunction(nnx.Module):

    def __init__(self, in_features: int, *, rngs: nnx.Rngs):
        self.q_function = MLP(in_features=in_features + TRIANGLE_SIZE, features=[128, 128, 1], activations=[nnx.selu, nnx.selu, None],
                              rngs=rngs)

    def __call__(self, core_output: Array, actions: DictArrayType) -> Array:
        triangle_action = actions["data"]["triangle"]
        features = jnp.concatenate([core_output, triangle_action], axis=1)
        return self.q_function(features)


class Policy(nnx.Module):

    def __init__(self, in_features: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.neck = MLP(in_features=in_features, features=[128, 128], activations=[nnx.selu, nnx.selu], rngs=rngs)
        self.mean_head = nnx.Linear(in_features=128, out_features=TRIANGLE_SIZE, rngs=rngs)
        self.log_std_head = nnx.Linear(in_features=128, out_features=TRIANGLE_SIZE, rngs=rngs)

    def __call__(self, core_output: Array) -> Tuple[Array, Array, Array]:
        neck_output = self.neck(core_output)
        means = self.mean_head(neck_output)
        log_std_dev = self.log_std_head(neck_output)
        std_dev = jnp.exp(log_std_dev)

        norm = distrax.MultivariateNormalDiag(loc=means, scale_diag=std_dev)
        dist = distrax.Transformed(
            distribution=norm, bijector=distrax.Block(distrax.Tanh(), ndims=1)
        )

        actions, action_log_prob = dist.sample_and_log_prob(seed=self.rngs())

        return actions, jnp.expand_dims(action_log_prob, -1), jnp.tanh(means)


if __name__ == "__main__":
    width = 300
    height = 600
    channels = 3
    env = TriangleEnv(width=width, height=height)
    image = jnp.zeros(shape=(height, width, channels))

    init_obs, _ = env.reset(options={"target": image})
    action = env.action_space.sample()

    rngs = nnx.Rngs(0)

    core = Core(rngs=rngs)
    q_function = QFunction(in_features=core.out_features, rngs=rngs)
    nnx.display(q_function)

    init_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), init_obs)
    action = jax.tree_map(lambda x: jnp.expand_dims(x, 0), action)

    core_output = core(init_obs)

    q = q_function(core_output=core_output, actions=action)
    print(q)

    policy = Policy(in_features=core.out_features, rngs=rngs)
    p, log_p, mean = policy(core_output=core_output)

    print(p, log_p, mean)
