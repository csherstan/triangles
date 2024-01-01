import os
os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import keras_core as keras
from keras import layers

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim:int, num_heads: int):
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, features, triangles, training):
        self.att(features, triangles)


if __name__=="__main__":
    pass