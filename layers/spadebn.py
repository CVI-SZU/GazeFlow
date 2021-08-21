#!/usr/bin/env python3

import tensorflow as tf

from layers.spectral_normalization import SpectralNormalization


class SpadeBN(tf.keras.layers.Layer):
    """SPADE BatchNormalization

    Sources:

        https://towardsdatascience.com/implementing-spade-using-fastai-6ad86b94030a
    """

    def __init__(self, width: int = 128, kernel_size=3, **kwargs):
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization()
        self.conv0 = SpectralNormalization(
            tf.keras.layers.Conv2D(width, kernel_size=kernel_size, activation="relu")
        )
        self.conv1 = SpectralNormalization(
            tf.keras.layers.Conv2D(width, kernel_size=kernel_size, activation="relu")
        )
        self.conv2 = SpectralNormalization(
            tf.keras.layers.Conv2D(width, kernel_size=kernel_size, activation="relu")
        )

    def call(self, x: tf.Tensor, cond: tf.Tensor):
        interim_conv = self.conv0(cond)
        gamma = self.conv1(interim_conv)
        beta = self.conv2(interim_conv)
        outputs = self.bn(x) * gamma + beta
        return outputs

    def get_config(self):
        config = super().get_config()
        config_update = {"width": self.width, "kernel_size": 3}
        config.update(config_update)
        return config
