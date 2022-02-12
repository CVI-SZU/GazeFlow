import numpy as np
import os
import pickle
import random
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from models.gazeflow import Glow


class GazeFlow:
    def __init__(self, hparams):
        self.hparams = hparams

        self.input_shape = [
            self.hparams.images_width,
            self.hparams.images_height,
            self.hparams.images_channel,
        ]
        self.condition_shape = self.hparams.condition_shape

        self.glow = Glow(hparams.K, hparams.L, hparams.conditional, hparams.width,
                         hparams.skip_type, condition_shape=self.condition_shape)
        self.setup_checkpoint((os.path.join(
            self.hparams.checkpoint_path, self.hparams.checkpoint_path_specific)))
#         self.check_model()

    def check_model(self):
        # self.glow.build(tuple([None] + self.input_shape))
        x = tf.keras.Input(self.input_shape)
        cond = tf.keras.Input(self.condition_shape)
        z, ldj, zaux, ll = self.glow(x, cond=cond, inverse=False)
        self.z_shape = list(z.shape)
        self.zaux_shape = list(zaux.shape)
        self.z_dims = np.prod(z.shape[1:])
        self.zaux_dims = np.prod(zaux.shape[1:])

        # summarize
        print("z_f's shape             : {}".format(self.z_shape))
        print("log_det_jacobian's shape: {}".format(ldj.shape))
        print("z_aux's shape           : {}".format(self.zaux_shape))
        self.glow.summary()

    def setup_checkpoint(self, checkpoint_path):
        print("checkpoint will be load at {}".format(checkpoint_path))
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), model=self.glow
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=7)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint has been restored !")

        self.ckpt = ckpt
