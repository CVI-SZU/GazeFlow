import numpy as np
import tensorflow as tf

from flows.inv1x1conv import Inv1x1Conv, Inv1x1Conv2DWithMask


class Inv1x1ConvTest(tf.test.TestCase):
    def setUp(self):
        super(Inv1x1ConvTest, self).setUp()
        self.inv1x1conv = Inv1x1Conv()
        self.inv1x1conv.build((None, 16, 16, 4))

    def testInv1x1ConvOutputShape(self):
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.inv1x1conv(x, inverse=False)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testInv1x1ConvOutput(self):
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.inv1x1conv(x, inverse=False)
        rev_x, ildj = self.inv1x1conv(z, inverse=True)
        self.assertAllClose(x, rev_x, rtol=1e-8, atol=1e-3)
        self.assertAllClose(ldj + ildj, tf.zeros([1024]), rtol=1e-8, atol=1e-8)


class Inv1x1Conv2DTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.inv1x1conv2D = Inv1x1Conv2DWithMask()
        self.inv1x1conv2D.build((None, None, 16))

    def testInv1x1Conv2DOutputShape(self):
        x = tf.random.normal([1024, 16, 16])
        z, ldj = self.inv1x1conv2D(x, inverse=False)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testInv1x1Conv2DOutput(self):
        x = tf.random.normal([1024, 16, 16])
        z, ldj = self.inv1x1conv2D(x, inverse=False)
        rev_x, ildj = self.inv1x1conv2D(z, inverse=True)
        self.assertAllClose(x, rev_x, rtol=1e-8, atol=1e-3)
        self.assertAllClose(ldj + ildj, tf.zeros([1024]), rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    tf.test.main(argv=None)
