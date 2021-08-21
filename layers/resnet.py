import tensorflow as tf
from tensorflow.keras import layers

from flows.utils.conv import Conv2D
from flows.utils.conv_zeros import Conv2DZeros
from layers.spadebn import SpadeBN

Layer = layers.Layer


def ShallowResNet(
    inputs: tf.keras.Input,
    cond: tf.keras.Input = None,
    width: int = 512,
    out_scale: int = 2,
):
    """ResNet of OpenAI's Glow

    Args:
        inputs (tf.Tensor): input tensor rank == 4
        cond   (tf.Tensor): input tensor rank == 4 (optional)
        width        (int): hidden width
        out_scale    (int): output channel width scale

    Returns:

        model: tf.keras.Model

    Sources:

        https://github.com/openai/glow/blob/master/model.py#L420-L426

    Note:

        This layer is not Residual Network
        because this layer does not have Skip connection


    Examples:

        >>> inputs = tf.keras.Input([16, 16, 2])
        >>> cond = None
        >>> sr = ShallowResNet(inputs)
        >>> sr.summary()
        Model: "model"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         [(None, 16, 16, 2)]       0
        _________________________________________________________________
        conv2d (Conv2D)              (None, 16, 16, 512)       10241
        _________________________________________________________________
        tf_op_layer_Relu (TensorFlow [(None, 16, 16, 512)]     0
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 16, 16, 512)       2360321
        _________________________________________________________________
        tf_op_layer_Relu_1 (TensorFl [(None, 16, 16, 512)]     0
        _________________________________________________________________
        conv2d_zeros (Conv2DZeros)   (None, 16, 16, 4)         18440
        =================================================================
        Total params: 2,389,002
        Trainable params: 2,389,000
        Non-trainable params: 2
        _________________________________________________________________
        >>> cond = tf.keras.Input([16, 16, 128])
        >>> sr = ShallowResNet(inputs, cond)
        >>> sr.summary()
        Model: "model_1"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_1 (InputLayer)            [(None, 16, 16, 2)]  0
        __________________________________________________________________________________________________
        input_3 (InputLayer)            [(None, 16, 16, 128) 0
        __________________________________________________________________________________________________
        tf_op_layer_concat_1 (TensorFlo [(None, 16, 16, 130) 0           input_1[0][0]
                                                                        input_3[0][0]
        __________________________________________________________________________________________________
        conv2d_2 (Conv2D)               (None, 16, 16, 512)  600065      tf_op_layer_concat_1[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_2 (TensorFlowO [(None, 16, 16, 512) 0           conv2d_2[0][0]
        __________________________________________________________________________________________________
        conv2d_3 (Conv2D)               (None, 16, 16, 512)  2360321     tf_op_layer_Relu_2[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_3 (TensorFlowO [(None, 16, 16, 512) 0           conv2d_3[0][0]
        __________________________________________________________________________________________________
        conv2d_zeros_1 (Conv2DZeros)    (None, 16, 16, 4)    18440       tf_op_layer_Relu_3[0][0]
        ==================================================================================================
        Total params: 2,978,826
        Trainable params: 2,978,824
        Non-trainable params: 2
        __________________________________________________________________________________________________

    """
    _inputs = inputs if cond is None else tf.concat([inputs, cond], axis=-1)

    conv1 = Conv2D(width=width)
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width=int(inputs.shape[-1] * out_scale))

    outputs = _inputs
    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = tf.keras.layers.ReLU()(conv2(outputs))
    outputs = conv_out(outputs)
    return tf.keras.Model(inputs if cond is None else [inputs, cond], outputs)


def ShallowConnectedResNet(
    inputs: tf.keras.Input,
    cond: tf.keras.Input = None,
    width: int = 512,
    out_scale: int = 2,
    connect_type: str = "whole",
):
    """ResNet of OpenAI's Glow with Connection

    Args:
        inputs (tf.Tensor): input tensor rank == 4
        cond   (tf.Tensor): input tensor rank == 4 (optional)
        width        (int): hidden width
        out_scale    (int): output channel width scale

    Returns:
        model: tf.keras.Model

    Sources:

        https://github.com/openai/glow/blob/master/model.py#L420-L426

    Examples:

        >>> inputs = tf.keras.Input([16, 16, 2])
        >>> cond = None
        >>> sr = ShallowConnectedResNet(inputs)
        >>> sr.summary()
        Model: "model"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_1 (InputLayer)            [(None, 16, 16, 2)]  0
        __________________________________________________________________________________________________
        conv2d_8 (Conv2D)               (None, 16, 16, 512)  10241       input_1[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_8 (TensorFlowO [(None, 16, 16, 512) 0           conv2d_8[0][0]
        __________________________________________________________________________________________________
        conv2d_9 (Conv2D)               (None, 16, 16, 512)  2360321     tf_op_layer_Relu_8[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_9 (TensorFlowO [(None, 16, 16, 512) 0           conv2d_9[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_concat (TensorFlowO [(None, 16, 16, 514) 0           tf_op_layer_Relu_9[0][0]
                                                                        input_1[0][0]
        __________________________________________________________________________________________________
        conv2d_zeros_4 (Conv2DZeros)    (None, 16, 16, 4)    18512       tf_op_layer_concat[0][0]
        ==================================================================================================
        Total params: 2,389,074
        Trainable params: 2,389,072
        Non-trainable params: 2
        __________________________________________________________________________________________________
        >>> cond = tf.keras.Input([16, 16, 128])
        >>> sr = ShallowResNet(inputs, cond, connect_type="cond")
        >>> sr.summary()
        sr.summary()
        Model: "model_4"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_1 (InputLayer)            [(None, 16, 16, 2)]  0
        __________________________________________________________________________________________________
        input_2 (InputLayer)            [(None, 16, 16, 128) 0
        __________________________________________________________________________________________________
        tf_op_layer_concat_3 (TensorFlo [(None, 16, 16, 130) 0           input_1[0][0]
                                                                        input_2[0][0]
        __________________________________________________________________________________________________
        conv2d_14 (Conv2D)              (None, 16, 16, 512)  600065      tf_op_layer_concat_3[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_14 (TensorFlow [(None, 16, 16, 512) 0           conv2d_14[0][0]
        __________________________________________________________________________________________________
        conv2d_15 (Conv2D)              (None, 16, 16, 512)  2360321     tf_op_layer_Relu_14[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_Relu_15 (TensorFlow [(None, 16, 16, 512) 0           conv2d_15[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_concat_4 (TensorFlo [(None, 16, 16, 640) 0           tf_op_layer_Relu_15[0][0]
                                                                        input_2[0][0]
        __________________________________________________________________________________________________
        conv2d_zeros_7 (Conv2DZeros)    (None, 16, 16, 4)    23048       tf_op_layer_concat_4[0][0]
        ==================================================================================================
        Total params: 2,983,434
        Trainable params: 2,983,432
        Non-trainable params: 2
        __________________________________________________________________________________________________
    """
    if cond is None and connect_type == "cond":
        raise ValueError("cond is none, but you want to connect with cond...")
    _inputs = inputs if cond is None else tf.concat([inputs, cond], axis=-1)

    conv1 = Conv2D(width=width)
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width=int(inputs.shape[-1] * out_scale))

    outputs = _inputs
    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = tf.keras.layers.ReLU()(conv2(outputs))
    # MokkeMeguru's skip-connection
    if connect_type == "whole":
        outputs = conv_out(tf.concat([outputs, _inputs], axis=-1))
    elif connect_type == "conditional":
        # conditional works too hevy work, so we help with skip-connection
        outputs = conv_out(tf.concat([outputs, cond], axis=-1))
    elif connect_type == "base":
        outputs = conv_out(tf.concat([outputs, inputs], axis=-1))
    else:
        outputs = conv_out(outputs)
    return tf.keras.Model(inputs if cond is None else [inputs, cond], outputs)


def ShallowConnectedResNetlikeSPADE(
    inputs: tf.keras.Input,
    cond: tf.keras.Input,
    width: int = 512,
    out_scale: int = 2,
    connect_type: str = "whole",
):
    """ResNet of OpenAI's Glow with Connection
    Note:

        WIP now...

    Args:
        inputs (tf.Tensor): input tensor rank == 4
        cond   (tf.Tensor): input tensor rank == 4 (optional)
        width        (int): hidden width
        out_scale    (int): output channel width scale

    Returns:
        model: tf.keras.Model

    Sources:

        https://github.com/openai/glow/blob/master/model.py#L420-L426
        https://github.com/NVlabs/SPADE

    Examples:

   """
    conv1 = Conv2D(width=width)
    spade1 = SpadeBN()
    conv2 = Conv2D(width=width)
    conv_out = Conv2DZeros(width=int(inputs.shape[-1] * out_scale))
    spade_conn = SpadeBN()

    outputs = inputs

    shortcut = outputs
    shortcut = spade_conn(shortcut, cond)

    outputs = tf.keras.layers.ReLU()(conv1(outputs))
    outputs = spade1(outputs, cond)
    outputs = tf.keras.layers.ReLU()(conv2(outputs))

    # MokkeMeguru's skip-connection
    outputs = conv_out(tf.concat([outputs, shortcut], axis=-1))

    return tf.keras.Model(inputs if cond is None else [inputs, cond], outputs)
