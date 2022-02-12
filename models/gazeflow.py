import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict

from flows import Actnorm, AffineCouplingMask, FlowModule, Inv1x1Conv
from flows.cond_affine_coupling import ConditionalAffineCoupling
from flows.factor_out import FactorOut, FactorOutBase
from flows.quantize import LogitifyImage
from flows.squeeze import Squeeze
from models.resnet import ConnectedResNet, ShallowConnectedResNet


class Glow(Model):
    def __init__(self, K, L, conditional=True, width=512, connect_type="whole", activation="softplus", condition_shape=(5,)):
        super().__init__()
        flows = []
        flows.append(LogitifyImage())

        for layer in range(L):
            # Squeeze Layer
            if layer == 0:
                flows.append(Squeeze(with_zaux=False))
            else:
                flows.append(Squeeze(with_zaux=True))
            fml = []
            # build flow module layer
            for k in range(K):
                fml.append(Actnorm())
                fml.append(Inv1x1Conv())
                # fml.append(Flip())
                cond = tf.keras.Input(condition_shape)
                fml.append(
                    ConditionalAffineCoupling(
                        scale_shift_net_template=lambda x: ConnectedResNet(
                            x, cond, width=width, connect_type=connect_type, activation=activation
                        ),
                    )
                )
            flows.append(FlowModule(fml))

            # Factor Out Layer
            if layer == 0:
                flows.append(FactorOut(conditional=conditional))
            elif layer != L - 1:
                flows.append(FactorOut(with_zaux=True,
                                       conditional=conditional))

        self.flows = flows

    def call(
        self,
        x: tf.Tensor,
        cond: tf.Tensor,
        zaux: tf.Tensor = None,
        inverse: bool = False,
        training: bool = True,
        temparature: float = 1.0,
    ):
        if inverse:
            return self.inverse(x, cond=cond, zaux=zaux, training=training, temparature=temparature)
        else:
            return self.forward(x, cond=cond, training=training)

    def inverse(
        self, x: tf.Tensor, cond: tf.Tensor, zaux: tf.Tensor, training: bool, temparature: float = 1.0
    ):
        """inverse
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])

        for flow in reversed(self.flows):
            if isinstance(flow, Squeeze):
                if flow.with_zaux:
                    if zaux is not None:
                        x, zaux = flow(x, zaux=zaux, inverse=True)
                    else:
                        x = flow(x, inverse=True)
                else:
                    x = flow(x, inverse=True)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True,
                                   temparature=temparature)
                else:
                    x = flow(x, zaux=zaux, inverse=True,
                             temparature=temparature)
            else:
                x, ldj = flow(x, cond=cond, inverse=True, training=training)
                inverse_log_det_jacobian += ldj
        return x, inverse_log_det_jacobian

    def forward(self, x: tf.Tensor, cond: tf.Tensor, training: bool, **kwargs):
        """forward
        object -> latent
        """
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeeze):
                if zaux is not None and flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
            elif isinstance(flow, FactorOutBase):
                x, zaux, ll = flow(x, zaux=zaux)
                log_likelihood += ll
            else:
                x, ldj = flow(x, cond=cond,  training=training)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux, log_likelihood
