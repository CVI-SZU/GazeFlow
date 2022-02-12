#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable, Dict

from flows.affine_coupling import AffineCouplingMask
from flows.flowbase import FlowComponent

Layer = layers.Layer
Conv2D = layers.Conv2D


def filter_kwargs(d: Dict):
    # utility function for Tensorflow's crasy constraint
    training = d.get("training", None)
    mask = d.get("mask", None)
    return {"training": training, "mask": mask}


class ConditionalAffineCoupling(FlowComponent):
    """Affine Coupling Layer

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN([x1, c])
            | scale = sigmoid(log_scale + 2.0)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN([z1, c])
            | scale = sigmoid(log_scale + 2.0)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
           | where c is the conditional input for WaveGlow or cINN
           | https://arxiv.org/abs/1907.02392

        * TODO notes
           | cINN uses double coupling, but our coupling is single coupling
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
        self.conditional_input = True
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

    def forward(self, x: tf.Tensor, cond: tf.Tensor, **kwargs):
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        h = self.scale_shift_net([x1, cond], **filter_kwargs(kwargs))
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

    def inverse(self, z: tf.Tensor, cond: tf.Tensor, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net([z1, cond], **filter_kwargs(kwargs))
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


class ConditionalAffineCouplingWithMask(ConditionalAffineCoupling):
    """Conditional Affine Coupling Layer with mask

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN([x1, c])
            | scale = exp(log_scale)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN([z1, c])
            | scale = exp(log_scale)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
           | where c is the conditional input for WaveGlow or cINN
           | https://arxiv.org/abs/1907.02392

        * TODO notes
           | cINN uses double coupling, but our coupling is single coupling
           |
           | scale > 0 because exp(x) > 0

        * mask notes
           | mask shape is [B, T, M] where M may be 1
           | reference glow-tts
    """

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        if self.scale_shift_net is None:
            resnet_inputs = [None for _ in range(len(input_shape) - 1)]
            resnet_inputs[-1] = int(input_shape[-1] / 2)
            self.scale_shift_net = self.scale_shift_net_template(
                tf.keras.layers.Input(resnet_inputs)
            )
        super().build(input_shape)

    def forward(self, x: tf.Tensor, cond: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        """
        Args:
            x    (tf.Tensor): base input tensor [B, T, C]
            cond (tf.Tensor): conditional input tensor [B, T, C']
            mask (tf.Tensor): mask input tensor [B, T, M] where M may be 1

        Returns:
            z    (tf.Tensor): latent variable tensor [B, T, C]
            ldj  (tf.Tensor): log det jacobian [B]

        Note:
            * mask's example
                | [[True, True, True, False],
                |  [True, False, False, False],
                |  [True, True, True, True],
                |  [True, True, True, True]]
        """
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        h = self.scale_shift_net([x1, cond], **filter_kwargs(kwargs))
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)

            # apply mask into scale, shift
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            if mask is not None:
                mask_tensor = tf.cast(mask, tf.float32)
                scale *= mask_tensor
                shift *= mask_tensor
            z2 = (x2 + shift) * scale

            # scale's shape is [B, T, C]
            # log_det_jacobian's shape is [B]
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
            return tf.concat([z1, z2], axis=-1), log_det_jacobian
        else:
            raise NotImplementedError()

    def inverse(self, z: tf.Tensor, cond: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net([x1, cond], **filter_kwargs(kwargs))
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)

            if mask is not None:
                mask_tensor = tf.cast(mask, tf.float32)
                scale *= mask_tensor
                shift *= mask_tensor
            x2 = (z2 / scale) - shift

            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
            return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
        else:
            raise NotImplementedError()


class ConditionalAffineCoupling2DWithMask(ConditionalAffineCouplingWithMask):
    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)
