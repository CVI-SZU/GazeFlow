import numpy as np
import tensorflow as tf

from flows.utils.conv_zeros import Conv2DZeros


class Conv2DTest(tf.test.TestCase):
    def setUp(self):
        super(Conv2DTest, self).setUp()
        self.conv2d = Conv2DZeros(width=None)
        self.conv2d_twice = Conv2DZeros(width_scale=2)
        self.conv2d.build((None, 16, 16, 4))
        self.conv2d_twice.build((None, 16, 16, 4))
        self.assertTrue(self.conv2d.built)
        self.assertTrue(self.conv2d_twice.built)

    def testConv2DOutputShape(self):
        x = tf.random.normal([512, 16, 16, 4])
        z = self.conv2d(x)
        self.assertShapeEqual(np.zeros(x.shape), z)

    def testConv2DOutputTwiceShape(self):
        x = tf.random.normal([512, 16, 16, 4])
        z_shape = list(tf.shape(x))
        z_shape[-1] = z_shape[-1] * 2
        z = self.conv2d_twice(x)
        self.assertShapeEqual(np.zeros(z_shape), z)


if __name__ == "__main__":
    tf.test.main(argv=None)
