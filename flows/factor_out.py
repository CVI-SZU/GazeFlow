import tensorflow as tf

from flows.flowbase import FactorOutBase
from flows.utils import gaussianize
from flows.utils.conv_zeros import Conv1DZeros, Conv2DZeros
from flows.utils.util import split_feature


class FactorOut(FactorOutBase):
    """Basic Factor Out Layer

    This layer drops factor-outed Tensor z_i

    Note:

        * forward procedure
           | input  : h_{i-1}
           | output : h_{i}, loss
           |
           | [z_i, h_i] = split(h_{i-1})
           |
           | loss =
           |     z_i \sim N(0, 1) if conditional is False
           |     z_i \sim N(mu, sigma) if conditional is True
           |  ,where
           | mu, sigma = Conv(h)

        * inverse procedure
           | input  : h_{i}
           | output : h_{i-1}
           |
           | sample z_i from N(0, 1) or N(mu, sigma) by conditional
           | h_{i-1} = [z_i, h_i]
    """

    def build(self, input_shape: tf.TensorShape):
        self.split_size = input_shape[-1] // 2
        super().build(input_shape)

    def __init__(
        self, with_zaux: bool = False, conditional: bool = False, dims: int = 4
    ):
        super().__init__()
        self.with_zaux = with_zaux
        self.conditional = conditional
        self.dims = dims
        if self.conditional:
            if dims == 4:
                self.conv = Conv2DZeros(width_scale=2)
            elif dims == 3:
                self.conv = Conv1DZeros(width_scale=2)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        if self.conditional:
            config_update = {
                "conditional": self.conditional,
                "conv": self.conv.get_config(),
                "dims": self.dims,
            }
        else:
            config_update = {"conditional": self.conditional}
        config.update(config_update)
        return config

    def split2d_prior(self, z: tf.Tensor):
        h = self.conv(z)
        return split_feature(h, "cross")

    def calc_ll(self, z1: tf.Tensor, z2: tf.Tensor):
        if self.conditional:
            mean, logsd = self.split2d_prior(z1)
            ll = gaussianize.gaussian_likelihood(mean, logsd, z2)
        else:
            ll = gaussianize.gaussian_likelihood(
                tf.zeros(tf.shape(z2)), tf.zeros(tf.shape(z2)), z2
            )
        ll = tf.reduce_sum(ll, axis=list(range(1, len(z2.shape))))
        return ll

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, **kwargs):
        new_z = x[..., : self.split_size]
        x = x[..., self.split_size :]

        ll = self.calc_ll(x, new_z)

        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux, ll

    def inverse(
        self, z: tf.Tensor, zaux: tf.Tensor = None, temparature: float = 0.2, **kwargs
    ):
        if zaux is not None:
            new_z = zaux[..., -self.split_size :]
            zaux = zaux[..., : -self.split_size]
        else:
            # TODO: sampling test
            mean, logsd = self.split2d_prior(z)
            new_z = gaussianize.gaussian_sample(mean, logsd, temparature)
        z = tf.concat([new_z, z], axis=-1)
        if self.with_zaux:
            return z, zaux
        else:
            return z


class FactorOutWithMask(FactorOut):
    """Basic Factor Out Layer With Mask

    This layer drops factor-outed Tensor z_i

    Note:

        * forward procedure
           | input  : h_{i-1}
           | output : h_{i}, loss
           |
           | [z_i, h_i] = split(h_{i-1})
           |
           | loss =
           |     z_i \sim N(0, 1) if conditional is False
           |     z_i \sim N(mu, sigma) if conditional is True
           |  ,where
           | mu, sigma = Conv(h)

        * inverse procedure
           | input  : h_{i}
           | output : h_{i-1}
           |
           | sample z_i from N(0, 1) or N(mu, sigma) by conditional
           | h_{i-1} = [z_i, h_i]

        * mask notes
            | mask shape is [B, T, M] where M may be 1
            | reference glow-tts
    """

    def calc_ll(self, z1: tf.Tensor, z2: tf.Tensor, mask_tensor: tf.Tensor = None):
        """
        Args:
           z1 (tf.Tensor): [B, T, C // 2]
           z2 (tf.Tensor): [B, T, C // 2]
        """
        with tf.name_scope("calc_log_likelihood"):
            if self.conditional:
                mean, logsd = self.split2d_prior(z1)
                ll = gaussianize.gaussian_likelihood(mean, logsd, z2)
            else:
                ll = gaussianize.gaussian_likelihood(
                    tf.zeros(tf.shape(z2)), tf.zeros(tf.shape(z2)), z2
                )
            # ll is [B, T, C // 2]
            if mask_tensor is not None:
                ll *= mask_tensor
                ll = tf.reduce_sum(ll, axis=list(range(1, len(z2.shape))))
            return ll

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, mask=None, **kwargs):
        if mask is not None:
            mask_tensor = tf.cast(mask, tf.float32)
        else:
            mask_tensor = None
        with tf.name_scope("split"):
            new_z = x[..., : self.split_size]
            x = x[..., self.split_size :]

        ll = self.calc_ll(x, new_z, mask_tensor=mask_tensor)

        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux, ll

    def inverse(
        self,
        z: tf.Tensor,
        zaux: tf.Tensor = None,
        mask=None,
        temparature: float = 0.2,
        **kwargs
    ):
        if zaux is not None:
            new_z = zaux[..., -self.split_size :]
            zaux = zaux[..., : -self.split_size]
        else:
            # TODO: sampling test
            mean, logsd = self.split2d_prior(z)
            new_z = gaussianize.gaussian_sample(mean, logsd, temparature)
        z = tf.concat([new_z, z], axis=-1)
        if self.with_zaux:
            return z, zaux
        else:
            return z


class FactorOut2DWithMask(FactorOutWithMask):
    def __init__(self, with_zaux: bool = False, conditional: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.with_zaux = with_zaux
        self.conditional = conditional
        if self.conditional:
            self.conv = Conv1DZeros(width_scale=2)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        if self.conditional:
            config_update = {
                "conditional": self.conditional,
                "conv": self.conv.get_config(),
            }
        else:
            config_update = {"conditional": self.conditional}
        config.update(config_update)
        return config

    def split2d_prior(self, z: tf.Tensor):
        h = self.conv(z)
        return split_feature(h, "cross")

    def calc_ll(self, z1: tf.Tensor, z2: tf.Tensor, mask_tensor: tf.Tensor = None):
        """Calculate log likelihood.
        Args:
           z1 (tf.Tensor): [B, T, C // 2]
           z2 (tf.Tensor): [B, T, C // 2]
        """
        with tf.name_scope("calc_log_likelihood"):
            if self.conditional:
                mean, logsd = self.split2d_prior(z1)
                ll = gaussianize.gaussian_likelihood(mean, logsd, z2)
            else:
                ll = gaussianize.gaussian_likelihood(
                    tf.zeros(tf.shape(z2)), tf.zeros(tf.shape(z2)), z2
                )
            # ll is [B, T, C // 2]
            if mask_tensor is not None:
                ll *= mask_tensor
                ll = tf.reduce_sum(ll, axis=list(range(1, len(z2.shape))))
            return ll

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, mask=None, **kwargs):
        if mask is not None:
            mask_tensor = tf.cast(mask, x.dtype)
        else:
            mask_tensor = None
        with tf.name_scope("split"):
            new_z = x[..., : self.split_size]
            x = x[..., self.split_size :]

        ll = self.calc_ll(x, new_z, mask_tensor=mask_tensor)

        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux, ll

    def inverse(
        self,
        z: tf.Tensor,
        zaux: tf.Tensor = None,
        mask=None,
        temparature: float = 0.2,
        **kwargs
    ):
        if zaux is not None:
            new_z = zaux[..., -self.split_size :]
            zaux = zaux[..., : -self.split_size]
        else:
            # TODO: sampling test
            mean, logsd = self.split2d_prior(z)
            new_z = gaussianize.gaussian_sample(mean, logsd, temparature)
        z = tf.concat([new_z, z], axis=-1)
        if self.with_zaux:
            return z, zaux
        else:
            return z


def main():
    layer = FactorOut()
    x = tf.random.normal([16, 4, 4, 128])
    y, zaux, ll = layer(x, zaux=None, inverse=False)
    _x = layer(y, zaux=zaux, inverse=True)
    print(x.shape)
    print(y.shape)
    print(zaux.shape)
    print(_x.shape)
    print(tf.reduce_sum(x - _x))
    layer = FactorOut(with_zaux=True)
    x = tf.random.normal([16, 8, 8, 8])
    zaux = tf.random.normal([16, 8, 8, 8])
    z, zaux, ll = layer(x, zaux=zaux, inverse=False)
    _x, _zaux = layer(z, zaux=zaux, inverse=True)
    print(x.shape)
    print(z.shape)
    print(zaux.shape)
    print(_x.shape)
    print(tf.reduce_mean(x - _x))
    layer = FactorOut()
    x = tf.keras.Input([8, 8, 8])
    z, zaux, ll = layer(x, zaux=None)
    model = tf.keras.Model(x, z)
    model.summary()
    return model
