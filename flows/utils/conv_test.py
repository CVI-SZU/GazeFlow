import numpy as np
import tensorflow as tf

from flows.utils.conv import Conv2D


class Conv2DTest(tf.test.TestCase):
    def setUp(self):
        super(Conv2DTest, self).setUp()
        self.conv2d = Conv2D(width=None, do_actnorm=True)
        self.conv2d_twice = Conv2D(width_scale=2, do_actnorm=True)
        self.conv2d.build((None, 16, 16, 4))
        self.conv2d_twice.build((None, 16, 16, 4))
        self.assertTrue(self.conv2d.built)
        self.assertTrue(self.conv2d_twice.built)

    def testConv2DOutputShape(self):
        x = tf.random.normal([512, 16, 16, 4])
        z = self.conv2d(x)
        self.assertTrue(self.conv2d.activation.initialized)
        self.assertShapeEqual(np.zeros(x.shape), z)

    def testConv2DOutputTwiceShape(self):
        x = tf.random.normal([512, 16, 16, 4])
        z_shape = list(tf.shape(x))
        z_shape[-1] = z_shape[-1] * 2
        z = self.conv2d_twice(x)
        self.assertTrue(self.conv2d_twice.activation.initialized)
        self.assertShapeEqual(np.zeros(z_shape), z)


if __name__ == "__main__":
    tf.test.main(argv=None)
