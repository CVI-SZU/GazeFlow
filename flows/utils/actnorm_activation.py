import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class ActnormActivation(Layer):
    """Actnorm Layer without inverse function

    This layer cannot sync mean / variance via Multi GPU

    Sources:

        https://github.com/openai/glow/blob/master/tfops.py#L71-L163

    Attributes:
        scale (float)          : scaling
        logscale_factor (float): logscale_factor

    Note:
        * initialize
            | mean = mean(first_batch)
            | var = variance(first-batch)
            | logs = log(scale / sqrt(var)) / log-scale-factor
            | bias = -mean

        * forward formula (forward only)
            | logs = logs * log_scale_factor
            | scale = exp(logs)
            | z = (x + bias) * scale
    """

    def __init__(self, scale: float = 1.0, logscale_factor=3.0, **kwargs):
        super(ActnormActivation, self).__init__()
        self.scale = scale
        self.logscale_factor = logscale_factor

    def get_config(self):
        config = super().get_config()
        config_update = {"scale": self.scale, "logscale_factor": self.logscale_factor}
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) == 4:
            reduce_axis = [0, 1, 2]
            b, h, w, c = list(input_shape)
        else:
            raise NotImplementedError()

        self.reduce_axis = reduce_axis

        stats_shape = [1 for _ in range(len(input_shape))]
        stats_shape[-1] = input_shape[-1]

        self.logs = self.add_weight(
            name="logscale",
            shape=tuple(stats_shape),
            initializer="zeros",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=tuple(stats_shape),
            initializer="zeros",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )

        self.initialized = self.add_weight(
            name="initialized",
            dtype=tf.bool,
            trainable=False,
            initializer=lambda shape, dtype: False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.build = True

    def data_dep_initialize(self, x: tf.Tensor):
        if self.initialized:
            bias, logs = self.bias, self.logs

        else:
            tf.print("initialization at {}".format(self.name))
            mean = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
            squared = tf.reduce_mean(tf.square(x), axis=[0, 1, 2], keepdims=True)

            variance = squared - tf.square(mean)

            bias = -mean
            logs = (
                tf.math.log(self.scale * tf.math.rsqrt(variance + 1e-6))
                / self.logscale_factor
            )

        with tf.control_dependencies([bias, logs]):
            self.bias.assign(bias)
            self.logs.assign(logs)
            self.initialized.assign(True)

    def call(self, x: tf.Tensor):

        self.data_dep_initialize(x)

        logs = self.logs * self.logscale_factor
        x = x + self.bias
        x = x * tf.exp(logs)
        return x


def main():
    aa = ActnormActivation()
    x = tf.keras.Input([16, 16, 2])
    y = aa(x)
    model = tf.keras.Model(x, y)
    model.summary()
    print(model.variables)
    y_ = model(tf.random.normal([32, 16, 16, 2]))
    print(y_.shape)
    print(model.variables)
