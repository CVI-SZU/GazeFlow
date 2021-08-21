#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from flows.quantize import LogitifyImage


class LogitifyImageTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.li = LogitifyImage()
        self.li.build([None, 32, 32, 1])

    def testLogitifyImageOutputShape(self):
        x = tf.nn.tanh(tf.random.normal([1024, 32, 32, 1]))
        z, ldj = self.li(x, inverse=False)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testLogitifyImageOutput(self):
        # x's range is [0, 1]
        x = tf.nn.sigmoid(tf.random.normal([1024, 32, 32, 1]))
        z, ldj = self.li(x, inverse=False)
        rev_x, ildj = self.li(z, inverse=True)
        self.assertAllClose(x, rev_x, rtol=8e-1, atol=1e-2)
        self.assertAllClose(ldj + ildj, tf.zeros([1024]))


def _main():
    layer = LogitifyImage()  # BasicGlow()
    x = tf.keras.Input((32, 32, 1))
    y = layer(x, training=True)
    model = tf.keras.Model(x, y)

    train, test = tf.keras.datasets.mnist.load_data()
    train_image = train[0] / 255.0
    train_image = train_image[..., tf.newaxis]
    # forward -> inverse
    train_image = train_image[0:12]
    forward, ldj = layer.forward(train_image)
    inverse, ildj = layer.inverse(forward)
    print(ldj)
    print(ildj)
    print(ldj + ildj)
    print(tf.reduce_mean(ldj + ildj))
    print(tf.reduce_mean(train_image - inverse))
    train_image = inverse
    print(train_image.shape)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 18))
    for i in range(9):
        img = tf.squeeze(train_image[i])
        fig.add_subplot(3, 3, i + 1)
        plt.title(train[1][i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.imshow(img, cmap="gray_r")
    plt.show(block=True)

    model.summary()
    return model
