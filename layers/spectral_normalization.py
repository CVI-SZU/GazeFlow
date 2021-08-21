#!/usr/bin/env python3
# reference: https://github.com/tensorflow/addons/pull/1244

import tensorflow as tf


class SpectralNormalization(tf.keras.layers.Wrapper):
    """This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm.

    Note:
        This stabilizes the training of GANs.
        Spectral Normalization for Generative Adversarial Networks:
        https://arxiv.org/abs/1802.05957
        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida (2018)
        SpectralNormalization wrapper works for keras and tf layers.

    Examples:

        >>> net = SpectralNormalization(
        >>>   tf.keras.layers.Conv2D(2, 2, activation="relu"),
        >>>     input_shape=(32, 32, 3))(x)
        >>> net = SpectralNormalization(
        >>>   tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
        >>> net = SpectralNormalization(
        >>>   tf.keras.layers.Dense(120, activation="relu"))(net)
        >>> net = SpectralNormalization(
        >>>   tf.keras.layers.Dense(n_classes))(net)

    Args:
      layer (tf.keras.layersLayer): a layer instance.

    Returns:
      tf.keras.layers.Layer: Wrapped Layer

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`
      AttributeError: If `Layer` does not contain a `kernel` or `embeddings` of weights
    """

    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False
        self._track_trackable(layer, name="layer")

    def build(self, input_shape):
        """Build `Layer`"""

        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if hasattr(self.layer, "kernel"):
                self.w = self.layer.kernel
            elif hasattr(self.layer, "embeddings"):
                self.w = self.layer.embeddings
            else:
                raise AttributeError(
                    "{} object has no attribute 'kernel' nor "
                    "'embeddings'".format(type(self.layer).__name__)
                )

            self.w_shape = self.w.shape.as_list()

            self.u = self.add_weight(
                shape=(1, self.w_shape[-1]),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                aggregation=tf.VariableAggregation.MEAN,
                trainable=False,
                name="sn_u",
                dtype=self.w.dtype,
            )

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of self.w with the
        spectral normalized value, so that the layer is ready for call().
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super(SpectralNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
