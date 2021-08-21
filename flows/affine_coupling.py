from enum import Enum
from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers

from flows.flowbase import FlowComponent

Layer = layers.Layer
Conv2D = layers.Conv2D


class LogScale(Layer):
    def build(self, input_shape: tf.TensorShape):
        shape = [1, input_shape[-1]]
        self.logs = self.add_weight(
            name="log_scale", shape=tuple(shape), initializer="zeros", trainable=True
        )

    def __init__(self, log_scale_factor: float = 3.0, **kwargs):
        super(LogScale, self).__init__(**kwargs)
        self.log_scale_factor = log_scale_factor

    def get_config(self):
        config = super().get_config()
        config_update = {"log_scale_factor": self.log_scale_factor}
        config.update(config_update)
        return config

    def call(self, x: tf.Tensor):
        return x * tf.exp(self.logs * self.log_scale_factor)


class AffineCouplingMask(Enum):
    ChannelWise = 1


class AffineCoupling(FlowComponent):
    """Affine Coupling Layer

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN(x1)
            | scale = sigmoid(log_scale + 2.0)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN(z1)
            | scale = sigmoid(log_scale + 2.0)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)

    Examples:

        >>> import tensorflow as tf
        >>> from flows.affine_coupling import AffineCoupling
        >>> from layers.resnet import ShallowResNet
        >>> af = AffineCoupling(scale_shift_net_template=ShallowResNet)
        >>> af.build([None, 16, 16, 4])
        >>> af.get_config()
            {'name': 'affine_coupling_1', ...}
        >>> inputs = tf.keras.Input([16, 16, 4])
        >>> af(inputs)
        (<tf.Tensor 'affine_coupling_3_2/Identity:0' shape=(None, 16, 16, 4) dtype=float32>,
         <tf.Tensor 'affine_coupling_3_2/Identity_1:0' shape=(None,) dtype=float32>)
        >>> tf.keras.Model(inputs, af(inputs)).summary()
        Model: "model_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_3 (InputLayer)         [(None, 16, 16, 4)]       0
        _________________________________________________________________
        affine_coupling (AffineCoupl ((None, 16, 16, 4), (None 2389003
        =================================================================
        Total params: 2,389,003
        Trainable params: 0
        Non-trainable params: 2,389,003
        _________________________________________________________________
    """

    def __init__(
        self,
        mask_type: AffineCouplingMask = AffineCouplingMask.ChannelWise,
        scale_shift_net: Layer = None,
        scale_shift_net_template: Callable[[tf.keras.Input], tf.keras.Model] = None,
        scale_type="safe_exp",
        **kwargs
    ):
        """
        Args:
            mask_type       (AffineCouplingMask: AffineCoupling Mask type
            scale_shift_net (tf.keras.Layer): NN in the fomula (Deprecated)
            scale_shift_net_template (Callable[[tf.keras.Input], [tf.keras.Model]]): NN in the formula (for tf.keras.Model without Input Shape)
        """
        super().__init__(**kwargs)
        if not scale_shift_net_template and not scale_shift_net:
            raise ValueError
        if scale_shift_net_template is not None:
            self.scale_shift_net_template = scale_shift_net_template
            self.scale_shift_net = None
        elif scale_shift_net is not None:
            self.scale_shift_net = scale_shift_net
        self.mask_type = mask_type

        self.scale_type = scale_type
        if self.scale_type not in ["safe_exp", "exp", "sigmoid"]:
            raise ValueError
        if self.scale_type == "safe_exp":
            self.scale_func = lambda log_scale: tf.exp(
                tf.clip_by_value(log_scale, -15.0, 15.0)
            )
        elif self.scale_type == "exp":
            self.scale_func = lambda log_scale: tf.exp(log_scale)
        else:
            self.scale_func = lambda log_scale: tf.nn.sigmoid(log_scale + 2.0)

    def get_config(self):
        config = super().get_config()
        config_update = {
            "scale_shit_net": self.scale_shift_net.get_config(),
            "mask_type": self.mask_type,
            "scale_type": self.scale_type,
        }
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        if self.scale_shift_net is None:
            resnet_inputs = list(input_shape)[1:]
            resnet_inputs[-1] = int(resnet_inputs[-1] / 2)
            self.scale_shift_net = self.scale_shift_net_template(
                tf.keras.Input(resnet_inputs)
            )

        super().build(input_shape)

    def forward(self, x: tf.Tensor, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        h = self.scale_shift_net(x1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)
            z2 = (x2 + shift) * scale

            # scale's shape is [B, H, W, C]
            # log_det_jacobian's hape is [B]
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
            return tf.concat([z1, z2], axis=-1), log_det_jacobian
        else:
            raise NotImplementedError()

    def inverse(self, z: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net(z1, **kwargs)
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)
            x2 = (z2 / scale) - shift

            # scale's shape is [B, H, W, C // 2]
            # inverse_log_det_jacobian's hape is [B]
            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
            return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
        else:
            raise NotImplementedError()
