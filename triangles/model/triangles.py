import flax.linen as nn
import jax.random
from jax import Array

# from triangles.common import DictArrayType

class CNN(nn.Module):

    @nn.compact
    def __call__(self, images: Array) -> Array:
        x = nn.Sequential([
            nn.Conv(features=64, kernel_sizes=(4,4), strides=(2,2), paddings="SAME", use_biases=False),
            nn.leaky_relu,
            nn.Conv(features=128, kernel_sizes=(4,4), strides=(2,2), paddings="SAME", use_biases=False),
            nn.leaky_relu,
            nn.Conv(features=256, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
            nn.leaky_relu,
            nn.Conv(features=512, kernel_sizes=(4, 4), strides=(2, 2), paddings="SAME", use_biases=False),
            nn.leaky_relu,
            nn.Conv(features=1, kernel_sizes=(4, 4), strides=(1, 1), paddings="VALID", use_biases=False),
            nn.leaky_relu,
        ])(images)

        return x

class QFunction(nn.Module):
    num_heads: int = 8
    num_attention_layers: int = 4
    qkv_features: int = 24

    @nn.compact
    def __call__(self) -> Array: #observations: DictArrayType, actions: DictArrayType) -> Array:
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
        """
        # error_inputs = observations["error"]
        # cnn_embedding = CNN()(error_inputs)
        #
        # op_one_hot = nn.one_hot(actions["op"], 3)

        # q, k, v = None, None, None
        q=jax.random.uniform(jax.random.key(0), shape=(1, 1, 24))
        k=jax.random.uniform(jax.random.key(0), shape=(1, 10, 24))

        for _ in range(self.num_attention_layers):
            attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_features)
            k_ = attention(inputs_q=q, inputs_k=k)
            k = nn.LayerNorm()(k+k_)

        return k



q = QFunction()
variables = q.init(jax.random.PRNGKey(0))
y = q.apply(variables)
print(y.shape)
