import numpy as np
import tensorflow as tf

from flows.actnorm import Actnorm


class ActnormTest(tf.test.TestCase):
    def setUp(self):
        super(ActnormTest, self).setUp()
        self.actnorm = Actnorm()
        self.actnorm.build((None, 16, 16, 4))

    def testActnormInitializeOutputShape(self):
        self.assertFalse(self.actnorm.initialized)
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.actnorm(x)
        self.assertTrue(self.actnorm.initialized)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testActnormOutput(self):
        x = tf.random.normal([1024, 16, 16, 4])
        self.actnorm(x)
        self.assertShapeEqual(np.zeros([1, 1, 1, 4]), self.actnorm.logs.value())
        self.assertShapeEqual(np.zeros([1, 1, 1, 4]), self.actnorm.bias.value())
        self.assertAllClose(
            self.actnorm.logs.value(),
            tf.math.log(
                self.actnorm.scale
                / tf.ones([1, 1, 1, 4])
                # / self.actnorm.logscale_factor
            ),
            # * self.actnorm.logscale_factor,
            rtol=1e-8,
            atol=1e-1,
        )
        self.assertAllClose(
            self.actnorm.bias.value(), tf.zeros([1, 1, 1, 4]), rtol=1e-8, atol=1e-1
        )


class ActnormScaleTest(tf.test.TestCase):
    def setUp(self):
        super(ActnormScaleTest, self).setUp()
        self.scale = 2.0
        self.actnorm = Actnorm(scale=self.scale)
        self.actnorm.build((None, 16, 16, 4))

    def testActnormInitializeOutputShape(self):
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.actnorm(x)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)


class ActnormInvTest(tf.test.TestCase):
    def setUp(self):
        super(ActnormInvTest, self).setUp()
        self.actnorm = Actnorm()
        self.actnorm.build((None, 16, 16, 4))

    def testActnormInitializeOutputShape(self):
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.actnorm(x)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testActnormInvOutput(self):
        x = tf.random.normal([1024, 16, 16, 4])
        self.actnorm(x)
        x = tf.random.normal([1024, 16, 16, 4])
        z, ldj = self.actnorm(x)
        rev_x, ildj = self.actnorm(z, inverse=True)
        self.assertAllClose(ldj + ildj, tf.zeros(x.shape[0:1]), rtol=1e-8, atol=1e-8)
        self.assertAllClose(x, rev_x, rtol=1e-8, atol=1e-5)


if __name__ == "__main__":
    tf.test.main(argv=None)
