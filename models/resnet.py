import tensorflow as tf
from flows.utils.conv import Conv2D
from flows.utils.conv_zeros import Conv2DZeros
from layers.spadebn import SpadeBN
from tensorflow.keras import layers

Layer = layers.Layer


def ShallowConnectedResNet(
    inputs: tf.keras.Input,
    cond: tf.keras.Input = None,
    width: int = 512,
    out_scale: int = 2,
    connect_type: str = "whole",
    activation: str = 'softplus'
):

    if cond is None and connect_type == "cond":
        raise ValueError("cond is none, but you want to connect with cond...")

    in_shape = list(inputs.shape[1:-1] + [1])
    modified_cond = tf.keras.layers.Dense(
        tf.math.reduce_prod(in_shape), activation=activation)(cond)
    modified_cond = tf.keras.layers.Reshape(in_shape)(modified_cond)
    _inputs = tf.keras.layers.Concatenate()([inputs, modified_cond])

    conv1 = Conv2D(width=width)
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width=int(inputs.shape[-1] * out_scale))

    outputs = _inputs
    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = tf.keras.layers.ReLU()(conv2(outputs))
    if connect_type == "whole":
        outputs = tf.keras.layers.Concatenate()([outputs, _inputs])
        outputs = conv_out(outputs)
    elif connect_type == "conditional":
        outputs = tf.keras.layers.Concatenate()([outputs, modified_cond])
        outputs = conv_out(outputs)
    elif connect_type == "base":
        outputs = tf.keras.layers.Concatenate()([outputs, inputs])
        outputs = conv_out(outputs)
    else:
        outputs = conv_out(outputs)
    return tf.keras.Model(inputs if cond is None else [inputs, cond], outputs)


def ConnectedResNet(
    inputs: tf.keras.Input,
    cond: tf.keras.Input = None,
    width: int = 512,
    out_scale: int = 2,
    connect_type: str = "whole",
    activation: str = 'softplus'
):

    if cond is None and connect_type == "cond":
        raise ValueError("cond is none, but you want to connect with cond...")

    in_shape = list(inputs.shape[1:-1] + [1])
    modified_cond = tf.keras.layers.Dense(
        tf.math.reduce_prod(in_shape) // 2, activation=activation)(cond)
    modified_cond = tf.keras.layers.Dense(tf.math.reduce_prod(
        in_shape), activation=activation)(modified_cond)
    modified_cond = tf.keras.layers.Reshape(in_shape)(modified_cond)
    _inputs = tf.keras.layers.Concatenate()([inputs, modified_cond])

    conv1 = Conv2D(width=width)
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width=int(inputs.shape[-1] * out_scale))

    outputs = _inputs
    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = tf.keras.layers.ReLU()(conv2(outputs))

    if connect_type == "whole":
        outputs = tf.keras.layers.Concatenate()([outputs, _inputs])
        outputs = conv_out(outputs)
    elif connect_type == "conditional":
        outputs = tf.keras.layers.Concatenate()([outputs, modified_cond])
        outputs = conv_out(outputs)
    elif connect_type == "base":
        outputs = tf.keras.layers.Concatenate()([outputs, inputs])
        outputs = conv_out(outputs)
    else:
        outputs = conv_out(outputs)
    return tf.keras.Model(inputs if cond is None else [inputs, cond], outputs)


def DenseConnectedResNet(
    xinputs: tf.keras.Input,
    cond: tf.keras.Input = None,
    width: int = 2046,
    out_scale: int = 2,
    connect_type: str = "whole",
    activation: str = 'softplus'
):

    if cond is None and connect_type == "cond":
        raise ValueError("cond is none, but you want to connect with cond...")

    in_shape = list(xinputs.shape[1:-1] + [1])
    real_inshape = list(xinputs.shape[1:-1] + [xinputs.shape[-1] * out_scale])
    real_real_inputshape = list(xinputs.shape[1:])
    modified_cond = tf.keras.layers.Dense(
        tf.math.reduce_prod(in_shape) // 4, activation=activation)(cond)
    modified_cond = tf.keras.layers.Dense(tf.math.reduce_prod(
        in_shape) // 2, activation=activation)(modified_cond)
    inputs = tf.keras.layers.Reshape(
        (tf.math.reduce_prod(real_real_inputshape).numpy(),))(xinputs)
    _inputs = tf.keras.layers.Concatenate()([inputs, modified_cond])

    conv1 = tf.keras.layers.Dense(width, kernel_initializer='ones')
    conv2 = tf.keras.layers.Dense(width*2, kernel_initializer='ones')
    conv3 = tf.keras.layers.Dense(width, kernel_initializer='ones')
    conv_out = tf.keras.layers.Dense(tf.math.reduce_prod(
        real_inshape).numpy(), kernel_initializer='ones', activation='sigmoid')

    outputs = _inputs
    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = tf.keras.layers.ReLU()(conv2(outputs))
    outputs = tf.keras.layers.ReLU()(conv3(outputs))

    if connect_type == "whole":
        outputs = tf.keras.layers.Concatenate()([outputs, _inputs])
        outputs = conv_out(outputs)
    elif connect_type == "conditional":
        outputs = tf.keras.layers.Concatenate()([outputs, modified_cond])
        outputs = conv_out(outputs)
    elif connect_type == "base":
        outputs = tf.keras.layers.Concatenate()([outputs, inputs])
        outputs = conv_out(outputs)
    else:
        outputs = conv_out(outputs)
    outputs = tf.keras.layers.Reshape(real_inshape)(outputs)
    return tf.keras.Model(inputs if cond is None else [xinputs, cond], outputs)


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = DenseConnectedResNet(
        tf.keras.Input((16, 32, 4)), tf.keras.Input((4,)))
    model.summary()
