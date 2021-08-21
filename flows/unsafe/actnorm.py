import tensorflow as tf

from flows.flowbase import FlowComponent


class Actnorm(FlowComponent):
    """Actnorm Layer

    This layer can SyncBatch Normalization, but may crash frequently.

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L71-L163

    Note:

        * initialize
            | mean = mean(first_batch)
            | var = variance(first_batch)
            | logs = log(scale / sqrt(var)) / logscale_factor
            | bias = - mean

        * forward formula
            | logs = logs * logscale_factor
            | scale = exp(logs)
            z = (x + bias) * scale
            log_det_jacobain = sum(logs) * H * W


        * inverse formula
            | logs = logs * logsscale_factor
            | inv_scale = exp(-logs)
            | z = x * inv_scale - bias
            | inverse_log_det_jacobian = sum(- logs) * H * W

    Attributes:
        calc_ldj: bool
            flag of calculate log det jacobian
        scale: float
            initialize batch's variance scaling
        logscale_factor: float
            barrier log value to - Inf
    """

    def __init__(self, scale: float = 1.0, logscale_factor: float = 3.0, **kwargs):
        super(Actnorm, self).__init__(**kwargs)
        self.scale = scale
        self.logscale_factor = logscale_factor

    def get_config(self):
        config = super().get_config()
        config_update = {"scale": self.scale, "logscale_factor": self.logscale_factor}
        config.update(config_update)
        return config

    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) == 4:
            reduce_axis = [0, 1, 2]
            b, h, w, c = list(input_shape)
            self.logdet_factor = tf.constant(h * w, dtype=tf.float32)
        else:
            raise NotImplementedError()

        self.reduce_axis = reduce_axis

        # stats_shape = [1, 1, 1, C] if input_shape == [B, H, W, C]
        stats_shape = [1 for _ in range(len(input_shape))]
        stats_shape[-1] = input_shape[-1]

        # self.logs = self.add_weight(
        #     name="logscale",
        #     shape=tuple(stats_shape),
        #     initializer="zeros",
        #     trainable=True,
        # )
        # self.bias = self.add_weight(
        #     name="bias", shape=tuple(stats_shape), initializer="zeros", trainable=True
        # )

        self.mean = self.add_weight(
            name="mean",
            shape=tuple(stats_shape),
            initializer="zeros",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self.squared = self.add_weight(
            name="squared",
            shape=tuple(stats_shape),
            initializer="zeros",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )

        super().build(input_shape)

    @property
    def bias(self):
        return -self.mean

    @property
    def logs(self):
        # var(x) = E(x^2) - E(x)^2
        variance = self.squared - tf.square(self.mean)
        logs = (
            #     # var(x) = E(x^2) - E(x)^2
            tf.math.log(self.scale * tf.math.rsqrt(variance + 1e-6))
            / self.logscale_factor
            #     variance = self.squared - tf.square(self.mean)
        )
        return logs

    def data_dep_initialize(self, x: tf.Tensor):

        if self.initialized:
            # bias, logs = self.bias, self.logs
            mean = self.mean
            squared = self.squared
        else:
            tf.print("initialization at {}".format(self.name))
            # mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
            # bias = -mean
            # logs = (
            #     #     # var(x) = E(x^2) - E(x)^2
            #     tf.math.log(self.scale * tf.math.rsqrt(variance + 1e-6))
            #     / self.logscale_factor
            #     #     variance = self.squared - tf.square(self.mean)
            # )
            mean = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
            squared = tf.reduce_mean(tf.square(x), axis=[0, 1, 2], keepdims=True)

        with tf.control_dependencies([mean, squared]):
            self.mean.assign(mean)
            self.squared.assign(squared)
            # self.bias.assign(bias)
            # self.logs.assign(logs)

            super().data_dep_initialize(x)

    def forward(self, x: tf.Tensor, **kwargs):
        logs = self.logs * self.logscale_factor

        # centering
        z = x + self.bias
        # scaling
        z = z * tf.exp(logs)

        log_det_jacobian = tf.reduce_sum(logs) * self.logdet_factor
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        logs = self.logs * self.logscale_factor

        # inverse scaling
        x = z * tf.exp(-1 * logs)
        # inverse centering
        x = x - self.bias

        inverse_log_det_jacobian = -1 * tf.reduce_sum(logs) * self.logdet_factor
        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(x)[0:1]
        )
        return x, inverse_log_det_jacobian
