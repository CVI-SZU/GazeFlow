from abc import ABCMeta, abstractmethod
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

Layer = layers.Layer


class FlowBase(Layer, metaclass=ABCMeta):
    """Flow-based model's abstruct class

    Examples:

        >>> layer = FlowBase()
        >>> z = layer(x, inverse=False) # forward method
        >>> x_hat = layer(z, inverse=True) # inverse method
        >>> assert tf.reduce_sum((x - x_hat)^2) << 1e-3

    Note:

        If you need data-dependent initialization (e.g. ActNorm),
        You can write it at #initialize_parameter.

         This layer will be inheritanced by the invertible layer without log det jacobian
    """

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conditional_input = False

    def data_dep_initialize(self, x: tf.Tensor):
        self.initialized.assign(True)

    def build(self, input_shape: tf.TensorShape):
        self.initialized = self.add_weight(
            name="initialized",
            dtype=tf.bool,
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            initializer=lambda shape, dtype: False,
        )
        self.built = True

    def call(self, x: tf.Tensor, inverse=False, **kwargs):
        if inverse and not self.initialized:
            raise Exception("Invalid initialize")

        self.data_dep_initialize(x)

        if inverse:
            return self.inverse(x, **kwargs)
        else:
            return self.forward(x, **kwargs)

    @abstractmethod
    def forward(self, inputs, **kwargs):
        return inputs

    @abstractmethod
    def inverse(self, inputs, **kwargs):
        return inputs


class FlowComponent(FlowBase):
    """Flow-based model's abstruct class

    Note:
         This layer will be inheritanced by the invertible layer with log det jacobian
    """

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x, **kwargs):
        log_det_jacobian = tf.zeros(x.shape[0:1])
        z = x
        return z, log_det_jacobian

    @abstractmethod
    def inverse(self, z, **kwargs):
        inverse_log_det_jacobian = tf.zeros(z.shape[0:1])
        x = z
        return x, inverse_log_det_jacobian

    def assert_tensor(self, x: tf.Tensor, z: tf.Tensor):
        if self.with_debug:
            tf.debugging.assert_shapes([(x, z.shape)])

    def assert_log_det_jacobian(self, log_det_jacobian: tf.Tensor):
        """assert log_det_jacobian's shape
       
        Note:
            | tf-2.0's bug
            | tf.debugging.assert_shapes([(tf.constant(1.0), (None, ))])
            | # => None (true)
            | tf.debugging.assert_shapes([(tf.constant([1.0, 1.0]), (None, ))])
            | # => None (true)
            | tf.debugging.assert_shapes([(tf.constant([[1.0], [1.0]]), (None, ))])
            | # => Error
        """
        if self.with_debug:
            tf.debugging.assert_shapes([(log_det_jacobian, (None,))])


class FlowModule(FlowBase):
    """Sequential Layer for FlowBase's Layer

    Examples:

         >>> layers = [FlowBase() for _ in range(10)]
         >>> module = FlowModule(layers)
         >>> z = module(x, inverse=False)
         >>> x_hat = module(z, inverse=True)
         >>> assert ((x - x_hat)^2) << 1e-3
    """

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape=input_shape)

    def __init__(self, components: List[FlowComponent], **kwargs):
        super().__init__(**kwargs)
        self.components = components

    def get_config(self):
        config = super().get_config()
        config_update_layer = []
        for comp in self.components:
            config_update_layer.append(comp.get_config())
        config_update = {"components", config_update_layer}
        config.update(config_update)
        return config

    def forward(self, x, **kwargs):
        z = x
        log_det_jacobian = []
        for component in self.components:
            z, ldj = component(z, inverse=False, **kwargs)
            log_det_jacobian.append(ldj)
        log_det_jacobian = sum(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z, **kwargs):
        x = z
        inverse_log_det_jacobian = []
        for component in reversed(self.components):
            x, ildj = component(x, inverse=True, **kwargs)
            inverse_log_det_jacobian.append(ildj)
        inverse_log_det_jacobian = sum(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


class ConditionalFlowModule(FlowBase):
    """Sequential Layer for FlowBase's Layer

    Examples:

         >>> layers = [FlowBase() for _ in range(10)]
         >>> module = FlowModule(layers)
         >>> z = module(x, inverse=False)
         >>> x_hat = module(z, inverse=True)
         >>> assert ((x - x_hat)^2) << 1e-3
    """

    def build(self, input_shape: tf.TensorShape):
        for component in self.components:
            component.build(input_shape)
        self.conditional_input = True    
        super().build(input_shape=input_shape)
        
    def __init__(self, components: List[FlowComponent], **kwargs):
        super().__init__(**kwargs)
        self.components = components

    def get_config(self):
        config = super().get_config()
        config_update_layer = []
        for comp in self.components:
            config_update_layer.append(comp.get_config())
        config_update = {"components", config_update_layer}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, cond: tf.Tensor, **kwargs):
        z = x
        log_det_jacobian = []
        for component in self.components:
            if component.conditional_input:
                z, ldj = component(z, inverse=False, cond=cond, **kwargs)
            else:
                z, ldj = component(z, inverse=False, **kwargs)
            log_det_jacobian.append(ldj)
        log_det_jacobian = sum(log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, cond: tf.Tensor, **kwargs):
        x = z
        inverse_log_det_jacobian = []
        for component in reversed(self.components):
            if component.conditional_input:
                x, ildj = component(x, inverse=True, cond=cond, **kwargs)
            else:
                x, ildj = component(x, inverse=True, **kwargs)
            inverse_log_det_jacobian.append(ildj)
        inverse_log_det_jacobian = sum(inverse_log_det_jacobian)
        return x, inverse_log_det_jacobian


class FactorOutBase(FlowBase):
    """Factor Out Layer in Flow-based Model

    Examples:

        >>> fo = FactorOutBase(with_zaux=False)
        >>> z, zaux = fo(x, zaux=None, inverse=False)
        >>> x = fo(z, zaux=zaux, inverse=True)

        >>> fo = FactorOutBase(with_zaux=True)
        >>> z, zaux = fo(x, zaux=zaux, inverse=False)
        >>> x, zaux = fo(z, zaux=zaux, inverse=True)
    """

    @abstractmethod
    def __init__(self, with_zaux: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.with_zaux = with_zaux

    def get_config(self):
        config = super().get_config()
        config_update = {"with_zaux": self.with_zaux}
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)

    def call(self, x: tf.Tensor, zaux: tf.Tensor = None, inverse=False, **kwargs):
        if inverse and not self.initialized:
            raise Exception("Invalid initialize")

        self.data_dep_initialize(x)

        if inverse:
            return self.inverse(x, zaux, **kwargs)
        else:
            return self.forward(x, zaux, **kwargs)

    @abstractmethod
    def forward(self, x: tf.Tensor, zaux: tf.Tensor, **kwargs):
        pass

    @abstractmethod
    def inverse(self, x: tf.Tensor, zaux: tf.Tensor, **kwargs):
        pass
