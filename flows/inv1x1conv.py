import numpy as np
import tensorflow as tf
from typing import Tuple

from flows.flowbase import FlowComponent


def regular_matrix_init(shape: Tuple[int, int], dtype=None):
    """initialize with orthogonal matrix

    Sources:
        https://github.com/openai/glow/blob/master/model.py#L445-L451

    Args:
        shape: generated matrix's shape [C, C]
        dtype:

    Returns:
       np.array: w_init, orthogonal matrix [C, C]

    """
    assert len(shape) == 2, "this initialization for 2D matrix"
    assert shape[0] == shape[1], "this initialization for 2D matrix, C \times C"
    c = shape[0]
    w_init = np.linalg.qr(np.random.randn(c, c))[0].astype("float32")
    # https://github.com/jaywalnut310/glow-tts/issues/17#issuecomment-644478717
    # https://github.com/NVIDIA/waveglow/blob/d18e0f3cc2ff6bdd41244d7391140accdc41142b/glow.py#L76
    # Ensure determinant is 1.0 not -1.0
    if np.linalg.det(w_init) < 0:
        w_init[:, 0] = -1 * w_init[:, 0]
    return w_init


class Inv1x1Conv(FlowComponent):
    """Invertible 1x1 Convolution Layer

    Sources:
        https://arxiv.org/pdf/1807.03039.pdf
        https://github.com/openai/glow/blob/master/model.py#L457-L472

    Note:
   
        * forward formula
            .. math::

                \\forall i, j: z_{i, j} &= Wx_{i, j} \\\\
                LogDetJacobian &= hw \log|det(W)|\\\\
                , where &\\\\
                W &\\in \\mathbb{R}^{c \times c}\\\\
                    x &\\in \\mathbb{R}^{b \\times h\\times w \\times c}\\ \\ \\
                ({\\rm batch, height, width, channel})

        * inverse formula
            .. math::

                \\forall i, j: x_{i, j} &= W^{-1} z_{i, j}\\\\
                InverseLogDetJacobian &= - h w \log|det(W)|\\\\
                , where &\\\\
                W &\\in \\mathbb{R}^{c\\times c}\\\\
                x &\\in \\mathbb{R}^{b \\times h\\times w \\times c}\\ \\ \\
                ({\\rm batch, height, width, channel})

    Examples:

        >>> import tensorflow as tf
        >>> from flows import Inv1x1Conv
        >>> ic = Inv1x1Conv()
        >>> ic.build([None, 16, 16, 4])
        >>> ic.get_config()
        {'name': 'inv1x1_conv_1', 'trainable': {}, 'dtype': 'float32'}
        >>> inputs = tf.keras.Input([16, 16, 4])
        >>> tf.keras.Model(inputs, ic(inputs)).summary()
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_3 (InputLayer)         [(None, 16, 16, 4)]       0
        _________________________________________________________________
        inv1x1_conv_1 (Inv1x1Conv)   ((None, 16, 16, 4), (None 17
        =================================================================
        Total params: 17
        Trainable params: 0
        Non-trainable params: 17
        _________________________________________________________________
    """

    def build(self, input_shape: tf.TensorShape):
        _, h, w, c = input_shape
        self.h = h
        self.w = w
        self.c = c
        self.W = self.add_weight(
            name="W",
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init,
        )
        super().build(input_shape)

    def __init__(self, log_det_type: str = "slogdet", **kwargs):
        super().__init__()
        self.log_det_type = log_det_type
        if self.log_det_type == "logdet":
            self.log_det_func = lambda x: tf.linalg.logdet(x)
        else:
            self.log_det_func = lambda x: tf.linalg.slogdet(x)[1]

    def get_config(self):
        config = super().get_config()
        config_update = {"log_det_type": self.log_det_type}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, **kwargs):
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(W, [1, 1, self.c, self.c])
        z = tf.nn.conv2d(x, _W, [1, 1, 1, 1], "SAME")
        # scalar
        log_det_jacobian = tf.cast(
            self.log_det_func(tf.cast(W, tf.float64)) * self.h * self.w, tf.float32,
        )
        # expand as batch
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(tf.linalg.inv(W), [1, 1, self.c, self.c])
        x = tf.nn.conv2d(z, _W, [1, 1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * self.log_det_func(tf.cast(W, tf.float64)) * self.h * self.w,
            tf.float32,
        )

        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(z)[0:1]
        )
        return x, inverse_log_det_jacobian


class Inv1x1Conv2DWithMask(FlowComponent):
    """Invertible 1x1 Convolution Layer (2D) with Mask

    Sources:
        https://arxiv.org/pdf/1807.03039.pdf
        https://github.com/openai/glow/blob/master/model.py#L457-L472

    Note:

        * forward formula
            .. math::

                \\forall i: z_{i} &= Wx_{i} \\\\
                LogDetJacobian &= t \log|det(W)|\\\\
                , where &\\\\
                W &\\in \\mathbb{R}^{c \times c}\\\\
                    x &\\in \\mathbb{R}^{b \\times h\\times w \\times c}\\ \\ \\
                ({\\rm batch, timestep, channel})

        * inverse formula
            .. math::

                \\forall i: x_{i} &= W^{-1} z_{i}\\\\
                InverseLogDetJacobian &= - t \log|det(W)|\\\\
                , where &\\\\
                W &\\in \\mathbb{R}^{c\\times c}\\\\
                x &\\in \\mathbb{R}^{b \\times t\\times c}\\ \\ \\
                ({\\rm batch, timestep, channel})

        * mask notes
            | mask shape is [B, T, M] where M may be 1
            | reference glow-tts
    """

    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        _, t, c = input_shape
        self.c = c
        self.W = self.add_weight(
            name="W",
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        """
        Args:
            x    (tf.Tensor): base input tensor [B, T, C]
            mask (tf.Tensor): mask input tensor [B, T, M] where M may be 1

        Returns:
            z    (tf.Tensor): latent variable tensor [B, T, C]
            ldj  (tf.Tensor): log det jacobian [B]

        Notes:
            * mask's example
                | [[[True], [True], [True], [False],
                |  [[True], [False], [False], [False],
                |  [[True], [True], [True], [True]],
                |  [[True], [True], [True], [True]]]
        """
        # b, t, c = tf.shape(x)
        shapes = tf.shape(x)
        b = shapes[0]
        t = shapes[1]
        c = shapes[2]

        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(W, [1, self.c, self.c])
        z = tf.nn.conv1d(x, _W, [1, 1, 1], "SAME")

        # scalar
        # tf.math.log(tf.abs(tf.linalg.det(W))) == tf.linalg.slogdet(W)[1]
        log_det_jacobian = tf.cast(
            tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        # expand as batch
        if mask is not None:
            mask_tensor = tf.cast(mask, tf.float32)
            z = z * mask_tensor
            # mask_tensor [B, T, M]
            log_det_jacobian = log_det_jacobian * tf.reduce_sum(
                tf.cast(mask, tf.float32), axis=[-2, -1]
            )
        else:
            log_det_jacobian = tf.broadcast_to(
                log_det_jacobian * tf.cast(t, tf.float32), tf.shape(x)[0:1]
            )
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        # b, t, c = tf.shape(x)
        shapes = tf.shape(z)
        b = shapes[0]
        t = shapes[1]
        c = shapes[2]

        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(tf.linalg.inv(W), [1, self.c, self.c])
        x = tf.nn.conv1d(z, _W, [1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        if mask is not None:
            mask_tensor = tf.cast(mask, tf.float32)
            x = x * mask_tensor
            inverse_log_det_jacobian = inverse_log_det_jacobian * tf.reduce_sum(
                tf.cast(mask, tf.float32), axis=[-2, -1]
            )
        else:
            inverse_log_det_jacobian = tf.broadcast_to(
                inverse_log_det_jacobian * tf.cast(t, tf.float32), tf.shape(z)[0:1]
            )
        return x, inverse_log_det_jacobian
