import numpy as np
import tensorflow as tf


def bits_x(
    log_likelihood: tf.Tensor, log_det_jacobian: tf.Tensor, pixels: int, n_bits: int = 8
):
    r"""bits/dims

    Sources:

        https://github.com/openai/glow/blob/master/model.py#L165-L186

    Args:
        log_likelihood (tf.Tensor): shape is [batch_size,]
        log_det_jacobian (tf.Tensor): shape is [batch_size,]
        pixels (int): e.g. HWC image => H * W * C
        n_bits (int): e.g [0 255] image => 8 = log(256)

    Returns:
        bits_x: shape is [batch_size,]

    Note:
        formula

        .. math::

          bits\_x = - \cfrac{(log\_likelihood + log\_det\_jacobian)}
          {pixels \log{2}} + n\_bits
    """
    nobj = -1.0 * (log_likelihood + log_det_jacobian)
    _bits_x = nobj / (np.log(2.0) * pixels) + n_bits
    return _bits_x


def split_feature(x: tf.Tensor, type: str = "split"):
    r"""type = [split, cross]

    TODO: implement Haar downsampling
    """
    channel = x.shape[-1]
    if type == "split":
        return x[..., : channel // 2], x[..., channel // 2 :]
    elif type == "cross":
        return x[..., 0::2], x[..., 1::2]
