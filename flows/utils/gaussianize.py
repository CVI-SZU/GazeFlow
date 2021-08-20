import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def gaussian_likelihood(mean: tf.Tensor, logsd: tf.Tensor, x: tf.Tensor):
    r"""calculate negative log likelihood of Gaussian Distribution.

    Args:
        mean  (tf.Tensor): mean [B, ...]
        logsd (tf.Tensor): log standard deviation [B, ...]
        x     (tf.Tensor): tensor [B, ...]

    Returns:
        ll    (tf.Tensor): log likelihood [B, ...]

    Note:

       .. math::
          :nowrap:

          \begin{align}
          ll &= - \cfrac{1}{2} (k\log(2 \pi) + \log |Var| \\
             &+ (x - Mu)^T (Var ^ {-1}) (x - Mu))\\
          ,\ where & \\
               & k = 1\ (Independent)\\
               & Var\ is\ a\ variance = exp(2 logsd)
          \end{align}
    """
    c = np.log(2 * np.pi)
    ll = -0.5 * (c + 2.0 * logsd + ((x - mean) ** 2) / tf.math.exp(2.0 * logsd))
    return ll


def gaussian_sample(mean: tf.Tensor, logsd: tf.Tensor, temparature: float = 1.0):
    r"""sampling from mean, logsd * temparature

    Args:
        mean (tf.Tensor): mean [B, ...]
        logsd(tf.Tensor): log standard deviation [B, ...]
        temparature      (float): temparature

    Returns:
        new_z(tf.Tensor): sampled latent variable [B, ...]

    Noto:
        I cann't gurantee it's correctness.
        Please open the tensorflow probability's Issue.
    """
    return tfp.distributions.Normal(mean, tf.math.exp(logsd) * temparature).sample()
