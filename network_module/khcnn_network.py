"""
Name:        "khcnn_network.py"
Author:      Marco Olivieri
Date:        23 November 2021
Description: KHCNN complete architecture

Note: this module comes from the KHCNN code by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834).
"""

import os
from utils import params as PARAMS

os.environ['CUDA_VISIBLE_DEVICES'] = PARAMS.GPUS
os.environ['CUDA_ALLOW_GROWTH'] = 'True'

import tensorflow as tf
from utils import data_processing as DATA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Lambda
from network_module import cnn_models as CNN
from network_module import kh_discretisation_tf as KH


def build_khcnn_model(p_size, s_size):
    """
    Build the KHCNN model
    :param p_size: pressure resolution
    :param s_size: surface resolution
    :return: keras model, losses dict, lossWeights dict
    """
    p_h_input_imgs = Input(shape=(p_size[0], p_size[1], 2))  # channels = real & imag of pressure
    freqs = Input(shape=(1,))                            # [None, 1]
    h_points = Input(shape=(p_size[0] * p_size[1], 3))  # [None, #points (1024 or 64), 3]
    s_points = Input(shape=(s_size[0] * s_size[1], 3))  # [None, 1024, 3]
    bm = Input(shape=(s_size[0], s_size[1], 2))

    # 2 outputs of shape=(16, 64, 2) - channles = real & imag
    # full structure KH
    if PARAMS.USE_LOW_P_RESOLUTION:
        v_s_output_imgs, p_s_output_imgs = CNN.cnn_model_64(p_h_input_imgs, start_neurons=4, out_channel=2,
                                                            name="complex_64")  # num = start_neurons
    else:
        v_s_output_imgs, p_s_output_imgs = CNN.cnn_model_1024(p_h_input_imgs, start_neurons=8, out_channel=2,
                                                              name="complex_1024")  # num = start_neurons

    v_s_output_imgs = Multiply(name="v_s_masked")([v_s_output_imgs, bm])
    p_s_output_imgs = Multiply(name="p_s_masked")([p_s_output_imgs, bm])

    # Use KHLayer to perform KH forward propagation inside a Keras Layer
    real_p_h_hat, imag_p_h_hat = KH.KHLayer()( [v_s_output_imgs, p_s_output_imgs, freqs, h_points, s_points] )

    # -----------------

    # loss on abs & phase
    v_s_hat = Lambda(lambda x: tf.complex(x[:, :, :, 0], x[:, :, :, 1]), name="v_s_complex")(v_s_output_imgs)
    p_h_hat = Lambda(lambda t: tf.complex(t[0], t[1]), name="p_h_complex")([real_p_h_hat, imag_p_h_hat])
    p_h_hat = Lambda(lambda x: DATA.normalise_wrt_abs_tf(x), name="norm_p_h")(p_h_hat)    # normalized wrt max(abs)

    real_p = Lambda(lambda x: tf.math.real(x), name="real_p_h")(p_h_hat)
    imag_p = Lambda(lambda x: tf.math.imag(x), name="imag_p_h")(p_h_hat)
    real_v = Lambda(lambda x: tf.math.real(x), name="real_v_s")(v_s_hat)
    imag_v = Lambda(lambda x: tf.math.imag(x), name="imag_v_s")(v_s_hat)

    model = Model(inputs=[p_h_input_imgs, freqs, h_points, s_points, bm],
                  outputs=[real_p, imag_p, real_v, imag_v])

    model.output_names = ["real_p_h_hat", "imag_p_h_hat",
                          "real_v_s_hat", "imag_v_s_hat"
                          ]

    losses = {
        "real_p_h_hat": "mean_squared_error",
        "imag_p_h_hat": "mean_squared_error",
        "real_v_s_hat": "mean_squared_error",
        "imag_v_s_hat": "mean_squared_error"
    }

    lossWeights = {
        "real_p_h_hat": 0.5, "imag_p_h_hat": 0.5,
        "real_v_s_hat": 0.5, "imag_v_s_hat": 0.5
    }

    return model, losses, lossWeights


# Compatibility alias: the violin test script builds the model through this name.
build_model_violin = build_khcnn_model