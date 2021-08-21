#!/usr/bin/env python3

import tensorflow as tf

from flows.squeeze import Squeeze, Squeeze2DWithMask


class Squeeze2DTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.squeeze = Squeeze2DWithMask()

    def testSqueezeWithOutAnythng(self):
        x = tf.random.normal([32, 16, 8])
        y, mask = self.squeeze(x, inverse=False)
        rev_x, mask = self.squeeze(y, inverse=True)
        self.assertAllEqual(x, rev_x)
        zaux = tf.random.normal([32, 16, 16])
        y, mask, new_zaux = self.squeeze(x, zaux=zaux, inverse=False)
        rev_x, mask, rev_zaux = self.squeeze(y, zaux=new_zaux, inverse=True)
        self.assertAllEqual(x, rev_x)
        self.assertAllEqual(zaux, rev_zaux)


class SqueezeTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.squeeze = Squeeze(with_zaux=True)

    def testSqueezeWithOutAnythng(self):
        x = tf.random.normal([32, 16, 16, 8])
        y = self.squeeze(x, inverse=False)
        rev_x = self.squeeze(y, inverse=True)
        self.assertAllEqual(x, rev_x)
        zaux = tf.random.normal([32, 16, 16, 12])
        y, new_zaux = self.squeeze(x, zaux=zaux, inverse=False)
        rev_x, rev_zaux = self.squeeze(y, zaux=new_zaux, inverse=True)
        self.assertAllEqual(x, rev_x)
        self.assertAllEqual(zaux, rev_zaux)
