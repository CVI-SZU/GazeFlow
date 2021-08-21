from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class Conv2DZeros(Layer):
    """Convolution layer for NHWC image with zero initialization
    Sources:
   
        https://github.com/openai/glow/blob/master/tfops.py#L292-L313

    Note:
        this layer not implemented.
        * function add_edge_padding
           ref. https://github.com/openai/glow/blob/master/tfops.py#L203-L232

        Xavier Initialize is better than other initializer

            ref. https://arxiv.org/pdf/1907.02392.pdf

            code.

                * https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal
                * https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform?version=nightly
    """

    def __init__(
        self,
        width: int = None,
        width_scale: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        logscale_factor: float = 3.0,
        initializer: tf.keras.initializers.Initializer = "zeros",
    ):
        super().__init__()
        self.width = width
        self.width_scale = width_scale
        self.kernel_size = list(kernel_size)
        self.stride = [1] + list(stride) + [1]
        self.padding = padding
        self.logscale_factor = logscale_factor
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config_update = {
            "width": self.width,
            "width_scale": self.width_scale,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "log_scale_factor": self.logscale_factor,
            "initializer": tf.keras.initializers.serialize(
                tf.keras.initializers.get(self.initializer)
            ),
        }
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        n_in = input_shape[-1]
        n_out = self.width if self.width is not None else n_in * self.width_scale
        filters = self.kernel_size + [n_in, n_out]
        self.kernel = self.add_weight(
            name="kernel", initializer=self.initializer, shape=filters, dtype=tf.float32
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[1 for _ in range(len(input_shape) - 1)] + [n_out],
            initializer="zeros",
        )
        self.logs = self.add_weight(
            name="logs", shape=[1, n_out], initializer=self.initializer
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor):
        x = tf.nn.conv2d(
            x, filters=self.kernel, strides=self.stride, padding=self.padding
        )
        x += self.bias
        x *= tf.exp(self.logs * self.logscale_factor)
        return x


class Conv1DZeros(tf.keras.layers.Layer):
    """Convolution layer for NTC text/audio with zero initialization

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L292-L313

    Note:
        this layer not implemented.
        * function add_edge_padding
           ref. https://github.com/openai/glow/blob/master/tfops.py#L203-L232

        Xavier Initialize is better than other initializer

            ref. https://arxiv.org/pdf/1907.02392.pdf

            code.

                * https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal
                * https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform?version=nightly

    Examples:

        >>> import tensorflow as tf
        >>> from flows.utils.conv_zeros import Conv1DZeros
        >>> c1z = Conv1DZeros(width_scale = 2)
        >>> x = tf.keras.layers.Input([None, 32]) # [B, T, C] where T is time-step and C is hidden-depth
        >>> y = clz(y) # [B, T, C * 2]
    """

    def __init__(
        self,
        width: int = None,
        width_scale: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "SAME",
        logscale_factor: float = 3.0,
        initializer: tf.keras.initializers.Initializer = "zeros",
    ):
        super().__init__()
        self.width = width
        self.width_scale = width_scale
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.logscale_factor = logscale_factor

    def get_config(self):
        config = super().get_config()
        config_update = {
            "width_scale": self.width_scale,
            "initializer": tf.keras.initializers.serialize(
                tf.keras.initializers.get(self.initializer)
            ),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        }
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        n_in = input_shape[-1]
        n_out = self.width if self.width is not None else n_in * self.width_scale
        self.kernel = self.add_weight(
            name="kernel",
            initializer=self.initializer,
            shape=[self.kernel_size] + [n_in, n_out],
            dtype=tf.float32,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[1 for _ in range(len(input_shape) - 1)] + [n_out],
            initializer="zeros",
        )
        self.logs = self.add_weight(
            name="logs", shape=[1, n_out], initializer=self.initializer
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor):
        x = tf.nn.conv1d(
            x, filters=self.kernel, stride=self.stride, padding=self.padding
        )
        x += self.bias
        x *= tf.exp(self.logs * self.logscale_factor)
        return x
