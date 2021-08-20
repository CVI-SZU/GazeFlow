#!/usr/bin/env python3

import tensorflow as tf

from flows.flowbase import FlowComponent


class Flatten(FlowComponent):
    """Flatten Layer
    Sources:

        https://github.com/VLL-HD/FrEIA/blob/26a5d4a901831a7f0130c6059b9d50ac72ae6f47/FrEIA/modules/reshapes.py#L204-L221

    Examples:

        >>> import tenosorflow as tf
        >>> from flows import Flatten
        >>> fl = Flatten()
        >>> fl.build([None, 16, 16, 2])
        >>> fl(inputs)
        (<tf.Tensor 'flatten_2_2/Identity:0' shape=(None, 512) dtype=float32>,
         <tf.Tensor 'flatten_2_2/Identity_1:0' shape=(None,) dtype=float32>)
        >>> tf.keras.Model(inputs, fl(inputs)).summary()
        Model: "model"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         [(None, 16, 16, 2)]       0
        _________________________________________________________________
        flatten_2 (Flatten)          ((None, 512), (None,))    1
        =================================================================
        Total params: 1
        Trainable params: 0
        Non-trainable params: 1
        _________________________________________________________________
        >>> z, ldj = fl(tf.random.normal([1024, 16, 16, 2]))
        >>> x, ildj = fl(z, invere=True)
        >>> x.shape
        TensorShape([1024, 16, 16, 2])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def get_config(self):
        return super().get_config()

    def build(self, input_shape):
        self.rebuild_shape = [-1] + list(input_shape)[1:]
        super().build(input_shape)

    def forward(self, x: tf.Tensor, **kwargs):
        z = self.flatten(x)
        log_det_jacobian = tf.zeros(shape=tf.shape(x)[0:1], dtype=tf.float32)
        return z, log_det_jacobian

    def inverse(self, x: tf.Tensor, **kwargs):
        z = tf.reshape(x, self.rebuild_shape)
        inverse_log_det_jacobian = tf.zeros(shape=tf.shape(x)[0:1], dtype=tf.float32)
        return z, inverse_log_det_jacobian
