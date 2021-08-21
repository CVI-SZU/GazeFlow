#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from flows.cond_affine_coupling import ConditionalAffineCoupling
from layers.resnet import ShallowResNet


class ConditionalAffineCouplingTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        cond = tf.keras.Input([32, 32, 16])
        self.cond_affine_coupling = ConditionalAffineCoupling(
            scale_shift_net_template=lambda inputs: ShallowResNet(
                inputs, cond=cond, width=512
            )
        )
        self.cond_affine_coupling.build([None, 32, 32, 128])

    def testConditonalAffineCouplingInitialize(self):
        assert not self.cond_affine_coupling.initialized
        x = tf.random.uniform((1024, 32, 32, 128))
        cond = tf.random.uniform((1024, 32, 32, 16))
        self.cond_affine_coupling(x, cond=cond, inverse=False)
        assert self.cond_affine_coupling.initialized

    def testConditionalAffineCouplingShape(self):
        x = tf.random.uniform((1024, 32, 32, 128))
        cond = tf.random.uniform((1024, 32, 32, 16))
        z, ldj = self.cond_affine_coupling(x, cond=cond, inverse=False)
        self.assertShapeEqual(np.zeros([1024, 32, 32, 128]), z)
        self.assertShapeEqual(np.zeros([1024]), ldj)

    def testConditionalAffineCouplingInv(self):
        x = tf.random.uniform((1024, 32, 32, 128))
        cond = tf.random.uniform((1024, 32, 32, 16))
        z, ldj = self.cond_affine_coupling(x, cond=cond, inverse=False)
        rev_x, ildj = self.cond_affine_coupling(x, cond=cond, inverse=True)
        self.assertAllClose(x, rev_x)
        self.assertAllClose(ldj + ildj, tf.zeros([1024]))
