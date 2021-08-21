from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from flows.utils.actnorm_activation import ActnormActivation

Layer = layers.Layer


class Conv2D(Layer):
    """Convolution layer for NHWC image

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L235-L264

    Note:
        this layer applies

        * data-dependent normalization (actnorm, openai's Glow)

        * weight normalization for stable training

        this layer not implemented.

        * function add_edge_padding

        ref. https://github.com/openai/glow/blob/master/tfops.py#L203-L232
    """

    def __init__(
        self,
        width: int = None,
        width_scale: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        do_actnorm: bool = True,
        do_weightnorm: bool = False,
        initializer: tf.keras.initializers.Initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05
        ),
        bias_initializer: tf.keras.initializers.Initializer = "zeros",
    ):
        super().__init__()
        self.width = width
        self.width_scale = width_scale
        self.kernel_size = list(kernel_size)
        self.stride = [1] + list(stride) + [1]
        self.padding = padding
        self.do_actnorm = do_actnorm
        self.do_weightnorm = do_weightnorm
        if self.do_actnorm:
            self.activation = ActnormActivation()
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config_update = {
            "width": self.width,
            "width_scale": self.width_scale,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "do_actnorm": self.do_actnorm,
            "do_weightnorm": self.do_weightnorm,
            "initializer": tf.keras.initializers.serialize(
                tf.keras.initializers.get(self.initializer)
            ),
        }
        if self.do_actnorm:
            config_update.update({"activation": self.activation.get_config()})
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        n_in = input_shape[-1]
        n_out = self.width if self.width is not None else n_in * self.width_scale
        filters = self.kernel_size + [n_in, n_out]
        self.kernel = self.add_weight(
            name="kernel", shape=filters, dtype=tf.float32, initializer=self.initializer
        )
        self.reduce_axis = list(range(len(input_shape) - 1))
        if not self.do_actnorm:
            self.bias = self.add_weight(
                name="bias",
                shape=[1 for _ in range(len(input_shape) - 1)] + [n_out],
                initializer="zeros",
            )
        self.built = True

    def call(self, x: tf.Tensor):
        if self.do_weightnorm:
            kernel = tf.nn.l2_normalize(self.kernel, self.reduce_axis)
        else:
            kernel = self.kernel

        x = tf.nn.conv2d(x, filters=kernel, strides=self.stride, padding=self.padding)
        if self.do_actnorm:
            x = self.activation(x)
        else:
            x += self.bias
        return x
