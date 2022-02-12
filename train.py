import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

import logging
from pathlib import Path
from typing import Dict
import argparse
import numpy as np
import tensorflow as tf
# tf.config.set_soft_device_placement(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
# 	tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_probability as tfp
from tensorflow.keras import metrics, optimizers
from optimizers import transformer_schedule
from flows.utils.util import bits_x
from models.model import Glow
from models.resnet import ConnectedResNet


# gpu growth constraint
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
AUTOTUNE = tf.data.experimental.AUTOTUNE

Mean = metrics.Mean
Adam = optimizers.Adam


class Gaze:
    def __init__(self, hparams):
        self.hparams = hparams
        self.input_shape = [
            self.hparams.images_width,
            self.hparams.images_height,
            self.hparams.images_channel,
        ]
        self.condition_shape = (self.hparams.condition_shape,)
        self.pixels = np.prod(self.input_shape)
        print(self.hparams.BATCH_SIZE)
        self.glow = Glow(
            hparams.K, 
            hparams.L, 
            hparams.conditional, 
            hparams.width, 
            hparams.skip_type, 
            condition_shape=self.condition_shape,
            scale_shift_net_template=ConnectedResNet)
        self.check_model()
        self.load_dataset()
        self.setup_target_distribution()
        self.setup_optimizer()
        self.setup_metrics()
        self.setup_checkpoint(Path(self.hparams.checkpoint_path, self.hparams.checkpoint_path_specific))
        self.setup_writer()
    
    def setup_writer(self):
        self.writer = tf.summary.create_file_writer(logdir=os.path.join(self.hparams.checkpoint_path, self.hparams.checkpoint_path_specific))

    def check_model(self):
        x = tf.keras.Input(self.input_shape)
        cond = tf.keras.Input(self.condition_shape)
        z, ldj, zaux, ll = self.glow(x, cond=cond, inverse=False)
        self.z_shape = list(z.shape)
        self.zaux_shape = list(zaux.shape)
        self.z_dims = np.prod(z.shape[1:])
        self.zaux_dims = np.prod(zaux.shape[1:])

        logger.info("z_f's shape             : {}".format(self.z_shape))
        logger.info("log_det_jacobian's shape: {}".format(ldj.shape))
        logger.info("z_aux's shape           : {}".format(self.zaux_shape))
        self.glow.summary()

    def load_dataset(self):
        print('Start load dataset.')
        raw_image_dataset = tf.data.TFRecordDataset(self.hparams.datapath)
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        }

        @tf.function
        def augument(example_proto):
            exp = tf.io.parse_single_example(example_proto, image_feature_description)
            img = tf.io.decode_jpeg(exp['image'])
            label = tf.io.parse_tensor(exp['label'], tf.float32)
            img = tf.cast(img, tf.float32)
            img = img / 255.0
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        total_train_batch = self.hparams.total_take//self.hparams.BATCH_SIZE
        raw_image_dataset = raw_image_dataset.map(augument, num_parallel_calls=AUTOTUNE).shuffle(self.hparams.total_take).batch(self.hparams.BATCH_SIZE)
        self.train_dataset = raw_image_dataset.take(total_train_batch)
        self.valid_dataset = raw_image_dataset.skip(total_train_batch).take(5)
        self.test_dataset = raw_image_dataset.skip(total_train_batch+10).take(2) 
        # self.train_dataset = self.train_dataset.shuffle(total_train_batch)
        # print(self.train_dataset)
        # print(self.valid_dataset)
        # print(self.test_dataset)
        # for i in self.train_dataset:
        #     print('1', i)
        #     break
        
        # for i in self.valid_dataset:
        #     print(2, i)
        #     break

        # for i in self.test_dataset:
        #     print(3, i)
        #     break

        # exit(0)
        print('Done')

    def setup_target_distribution(self):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.ones([self.z_dims])
        )
        zaux_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.zaux_dims]), tf.ones([self.zaux_dims])
        )
        self.target_distribution = (z_distribution, zaux_distribution)

    def setup_optimizer(self):
        self.learning_rate_schedule = transformer_schedule.CustomSchedule(self.pixels * 20.)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate_schedule)
        # self.optimizer = tf.keras.optimizers.SGD(self.learning_rate_schedule)

    def setup_metrics(self):
        self.train_nll = Mean(name="b/d", dtype=tf.float32)
        self.valid_nll = Mean(name="b/d", dtype=tf.float32)

    def setup_checkpoint(self, checkpoint_path: Path):
        logger.info("checkpoint'path : {}".format(checkpoint_path))
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), model=self.glow, optimizer=self.optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=7)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            logger.info("Checkpoint restored")
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    def encode(self, img, cond):
        """
        Input:
            img  : Input images.
            cond : Input images' corresponding conditions.
        Output:
            z    : Downsampled latent code.
            zaux : Latent code that is split-out by split layer.
        """
        if len(img.shape)==3:
            img = tf.expand_dims(img, axis=0)
        if len(cond.shape)==1:
            cond = tf.expand_dims(cond, axis=0)
        assert cond.shape[0] == img.shape[0]
        z, _, zaux, _ = self.glow.forward(img, cond, training=False)
        return z, zaux

    def decode(self, latent, cond, zaux):
        """
        Input:
            latent  : Input latent codes z.
            cond    : Input images' corresponding conditions.
            zaux    : Latent code that is split-out by split layer, keep it for better reconstruction
        Output:
            Image with condition cond.
        """
        if len(latent.shape)==3:
            latent = tf.expand_dims(latent, axis=0)
        if len(cond.shape)==1:
            cond = tf.expand_dims(cond, axis=0)
        assert cond.shape[0] == latent.shape[0]
        assert zaux.shape[:-1] == latent.shape[:-1]
        return self.glow.inverse(latent, cond,  zaux = zaux, training=False)[0]

    def sample_image(self, beta_z: float = 0.75, beta_zaux: float = 0.75):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.broadcast_to(beta_z, [self.z_dims])
        )
        z = z_distribution.sample(self.hparams.BATCH_SIZE)
        z = tf.reshape(z, [-1] + self.z_shape[1:])
        self.valid_dataset = self.valid_dataset.shuffle(10*self.hparams.BATCH_SIZE)
        for td in self.valid_dataset.take(1):
            cond = td[1]
        x, ildj = self.glow.inverse(z, cond=cond, zaux=None, training=False, temparature=beta_zaux)
        x = tf.clip_by_value(x, 0.0, 1.0)
        tf.summary.image(
            "generated image", x, step=self.optimizer.iterations, max_outputs=4
        )
        for x in self.valid_dataset.take(1):
            tf.summary.image(
                "original image",
                x[0][:4],
                max_outputs=4,
                step=self.optimizer.iterations,
            )
            z, ldj, zaux, ll = self.glow(x[0][:4], cond=x[1][:4], training=False)
            x, ildj = self.glow.inverse(z, x[1][5:9], zaux, training=False, temparature=1.0)
            x = tf.clip_by_value(x, 0.0, 1.0)
            tf.summary.image(
                "conditional edited image", x, max_outputs=4, step=self.optimizer.iterations
            )


    @tf.function
    def train_step(self, img):
        with tf.GradientTape() as tape:
            z, ldj, zaux, ll = self.glow(img[0], cond=img[1], training=True)
            z = tf.reshape(z, [-1, self.z_dims])
            zaux = tf.reshape(zaux, [-1, self.zaux_dims])
            lp = self.target_distribution[0].log_prob(z)
            loss = bits_x(lp + ll, ldj, self.pixels)
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        self.train_nll(loss)
        return tf.reduce_mean(loss)

    def train(self):
        for epoch in range(self.hparams.epochs):
            count = 0
            for x in self.train_dataset:
                count += 1
                loss = self.train_step(x)
                print(
                        "loss:", loss,
                        ",iter:", count,
                        ",epoch", epoch,
                        "****", end='\r'
                    )
            ckpt_save_path = self.ckpt_manager.save()
            with self.writer.as_default():
                self.sample_image(
                    self.hparams.beta_z,
                    self.hparams.beta_zaux,
                )
                # print('finished sampling...')
                tf.summary.scalar(
                    "train/nll", self.train_nll.result(), step=self.optimizer.iterations
                )
                tf.summary.scalar(
                    "valid/nll", self.valid_nll.result(), step=self.optimizer.iterations
                )
                # tf.summary.scalar(
                #     "train/ldj", self.train_ldj.result(), step=self.optimizer.iterations
                # )
                # tf.summary.scalar(
                #     "valid/ldj", self.valid_ldj.result(), step=self.optimizer.iterations
                # )
                logger.info(
                    "epoch {}: train_loss = {}, valid_loss = {}, saved_at = {}".format(
                        epoch,
                        self.train_nll.result().numpy(),
                        self.valid_nll.result().numpy(),
                        ckpt_save_path,
                    )
                )
                self.train_nll.reset_states()
                # self.train_ldj.reset_states()
                self.valid_nll.reset_states()
                # self.valid_ldj.reset_states()
            


if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--BATCH-SIZE', type=int, default=64,  help='training batch size')
    parser.add_argument('--total-take', type=int, default=64,  help='total size of training data')
    parser.add_argument('--images-width', type=int, default=64,  help='width')
    parser.add_argument('--images-height', type=int, default=32,  help='height')
    parser.add_argument('--images-channel', type=int, default=3,  help='channels')
    parser.add_argument('--K', type=int,  default=18, help='k steps of flow-step')
    parser.add_argument('--L', type=int,  default=3, help='L levels of multiscale level')
    parser.add_argument('--conditional', type=bool, default=True,  help='split layer constraint')
    parser.add_argument('--width', type=int, default=256,  help='condition affine coupling net width')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints',  help='route to checkpoints')
    parser.add_argument('--epochs', type=int, default=100,  help='training epochs')

    parser.add_argument('--beta_z', type=float, default=0.75,  help='sampling parameters')
    parser.add_argument('--beta_zaux', type=float, default=0.75,  help='sampling parameters')

    parser.add_argument('--condition-shape', type=int, default=5,  help='number of components in condition')
    parser.add_argument('--skip-type', type=str, default='whole', help='parameters of condition encoder')
    parser.add_argument('--checkpoint-path-specific', type=str, default='test',  help='checkpoints folder')
    parser.add_argument('--datapath', type=str, required=True,  help='data folder')

    args = parser.parse_args()
    # hparams = Hparams()
    gaze = Gaze(args)
    # train...
    gaze.train()



















