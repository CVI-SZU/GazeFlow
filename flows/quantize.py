import numpy as np
import tensorflow as tf

from flows.flowbase import FlowBase


class LogitifyImage(FlowBase):
    """Apply Tapani Raiko's dequantization and express image in terms of logits

    Sources:

        https://github.com/taesungp/real-nvp/blob/master/real_nvp/model.py
        https://github.com/taesungp/real-nvp/blob/master/real_nvp/model.py#L42-L54
        https://github.com/tensorflow/models/blob/fe4e6b653141a197779d752b422419493e5d9128/research/real_nvp/real_nvp_multiscale_dataset.py#L1073-L1077
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/operations.py#L253-L254
        https://github.com/fmu2/realNVP/blob/8d36691df215af3678440ccb7c01a13d2b441a4a/data_utils.py#L112-L119

    Args:
        corrupution_level (float): power of added random variable.
        alpha (float)            : parameter about transform close interval to open interval
                                   [0, 1] to (1, 0)

    Note:

        We know many implementation on this quantization, but we use this formula.
        since many implementations use it.

        * forward preprocess (add noise)
            .. math::

                z &\\leftarrow 255.0 x  \\ \\because [0, 1] \\rightarrow [0, 255] \\\\
                z &\\leftarrow z + \\text{corruption_level} \\times  \\epsilon \\ where\\ \\epsilon \\sim N(0, 1)\\\\
                z &\\leftarrow z / (\\text{corruption_level} + 255.0)\\\\
                z &\\leftarrow z (1 - \\alpha) + 0.5 \\alpha \\ \\because \\ [0, 1] \\rightarrow (0, 1) \\\\
                z &\\leftarrow \log(z) - \log(1 -z)

        * forward formula
            .. math::

                 z &= logit(x (1 - \\alpha) + 0.5 \\alpha)\\\\
                   &= \\log(x) - \\log(1 - x)\\\\
                 LogDetJacobian &= sum(softplus(z) + softplus(-z) - softplus(\\log(\\cfrac{\\alpha}{1 - \\alpha})))

        * inverse formula
            .. math::

                 x &= logisitic(z)\\\\
                   &= 1 / (1 + exp( -z )) \\\\
                 x &= (x - 0.5 * \\alpha) / (1.0 -  \\alpha)\\\\
                 InverseLogDetJacobian &= sum(-2 \\log(logistic(z)) - z) + softplus(\\log(\\cfrac{\\alpha}{1 - \\alpha})))
    Examples:

        >>> import tensorflow as tf
        >>> from flows import LogitifyImage
        >>> li = LogitifyImage()
        >>> li.build([None, 32, 32, 1])
        >>> li.get_config()
        {'name': 'logitify_image_1', ...}
        >>> inputs = tf.keras.Input([32, 32, 1])
        >>> li(inputs)
        (<tf.Tensor 'logitify_image/Identity:0' shape=(None, 32, 32, 1) dtype=float32>,
        <tf.Tensor 'logitify_image/Identity_1:0' shape=(None,) dtype=float32>)
        >>> tf.keras.Model(inputs, li(inputs)).summary()
        Model: "model"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         [(None, 32, 32, 1)]       0
        _________________________________________________________________
        logitify_image (LogitifyImag ((None, 32, 32, 1), (None 1
        =================================================================
        Total params: 1
        Trainable params: 0
        Non-trainable params: 1
        _________________________________________________________________
    """

    def build(self, input_shape: tf.TensorShape):
        super(LogitifyImage, self).build(input_shape)
        # ref. https://github.com/masa-su/pixyz/blob/master/pixyz/flows/operations.py#L254
        self.pre_logit_scale = tf.constant(
            np.log(self.alpha) - np.log(1.0 - self.alpha), dtype=tf.float32
        )
        if len(input_shape) == 4:
            self.reduce_axis = [1, 2, 3]
        elif len(input_shape) == 2:
            self.reduce_axis = [1]
        else:
            raise NotImplementedError()

        super(LogitifyImage, self).build(input_shape)

    def __init__(self, corruption_level=1.0, alpha=0.05):
        super(LogitifyImage, self).__init__()
        self.corruption_level = corruption_level
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        config_update = {"corruption_level": self.corruption_level, "alpha": self.alpha}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, **kwargs):
        """
        """

        # 1. transform the domain of x from [0, 1] to [0, 255]
        z = x * 255.0

        # 2-1. add noize to pixels to dequantize them
        # and transform its domain ([0, 255]->[0, 1])
        z = z + self.corruption_level * tf.random.uniform(tf.shape(x))
        z = z / (255.0 + self.corruption_level)

        # 2-2. transform pixel values with logit to be unconstrained
        # ([0, 1]->(0, 1)).
        # TODO: Will this function polutes the image?
        z = z * (1 - self.alpha) + self.alpha * 0.5

        # 2-3. apply the logit function ((0, 1)->(-inf, inf)).
        z = tf.math.log(z) - tf.math.log(1.0 - z)

        logdet_jacobian = (
            tf.math.softplus(z)
            + tf.math.softplus(-z)
            - tf.math.softplus(self.pre_logit_scale)
        )

        logdet_jacobian = tf.reduce_sum(logdet_jacobian, self.reduce_axis)
        return z, logdet_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        """
        """
        denominator = 1 + tf.exp(-z)
        x = 1 / denominator

        x = (x - 0.5 * self.alpha) / (1.0 - self.alpha)

        inverse_log_det_jacobian = -1 * (
            tf.math.softplus(z)
            + tf.math.softplus(-z)
            - tf.math.softplus(self.pre_logit_scale)
        )

        inverse_log_det_jacobian = tf.reduce_sum(
            inverse_log_det_jacobian, self.reduce_axis
        )

        return x, inverse_log_det_jacobian
