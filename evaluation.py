import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from model import Glow
from layers import ConnectedResNet
import argparse
from pathlib import Path
from PIL import Image

# hyper parameters
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--K', type=int,  default=18, help='k steps of flow-step')
parser.add_argument('--L', type=int,  default=3, help='L levels of multiscale level')
parser.add_argument('--conditional', type=bool, default=True,  help='split layer constraint')
parser.add_argument('--width', type=int, default=256,  help='condition affine coupling net width')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints',  help='route to checkpoints')
parser.add_argument('--condition-shape', type=int, default=5,  help='number of components in condition')
parser.add_argument('--skip-type', type=str, default='whole', help='parameters of condition encoder')
parser.add_argument('--checkpoint-path-specific', type=str, default='test',  help='checkpoints folder')

args = parser.parse_args()
gazeflow = Glow(
            args.K, 
            args.L, 
            args.conditional, 
            args.width, 
            args.skip_type, 
            condition_shape=(args.condition_shape,),
            scale_shift_net_template=ConnectedResNet)
ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), model=gazeflow, optimizer=tf.keras.optimizers.Adam(1e-4)
        )
ckpt_manager = tf.train.CheckpointManager(ckpt, Path(args.checkpoint_path, args.checkpoint_path_specific), max_to_keep=7)
ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()


def encode(img, cond):
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
    z, _, zaux, _ = gazeflow.forward(img, cond, training=False)
    return z, zaux

def decode(latent, cond, zaux):
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
    return gazeflow.inverse(latent, cond,  zaux = zaux, training=False)[0]

"""
conditions for image : '1.png' : [[ 0.5826192 , -0.61143166,  0.5749567 , -0.2196801 ,  1.        ]]
conditions for image : '2.png' : [[-0.7409178 , -0.09921838, -0.49584487, -0.12268476,  1.        ]]
"""
image = np.array(Image.open('./imgs/1.png')).astype(np.float32)/255.
condition = tf.convert_to_tensor([[ 0.5826192 , -0.61143166,  0.5749567 , -0.2196801 ,  1.        ]]) # Gpitch, Gyaw, Hpitch, Hyaw, Eside

modify = tf.convert_to_tensor([[ 0.2 , -0.5,  0. , 0. ,  0.   ]]) # Shift of each components. `0.2` means `Gpitch + 0.2`

z, zaux = encode(image, condition)
Image.fromarray(np.clip(decode(z, condition + modify, zaux)*255, 0., 255.).astype(np.uint8)).save('./imgs/1_modified.png')






