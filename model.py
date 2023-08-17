# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.initializers import Initializer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from data import image_size_dict
from secondpooling import SecondOrderPooling

from keras.utils import np_utils
import tensorflow as tf
import keras
#from tensorflow.keras.layers import InputSpec
from keras import activations
from keras.layers import Permute
from pandas.core.frame import DataFrame
from keras.engine.base_layer import Layer
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import seaborn as sns
import time
from functions import *

from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, concatenate, activations, merge, add, Multiply

def cal_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('flops'+ str(flops.total_float_ops))
    print('params'+ str(params.total_parameters))

def PFM(Input):
    x=Input
    x1 = Conv2D(filters=64, kernel_size=11, activation='relu', strides=2, padding='same')(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(filters=64, kernel_size=7, activation='relu', strides=2, padding='same')(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x3 = BatchNormalization()(x3)
    
    x4 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x4 = BatchNormalization()(x4)

    x5 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x5 = BatchNormalization()(x5)

    x6 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x6 = BatchNormalization()(x6)
    
    ms = concatenate([x1, x2, x3, x4, x5, x6])
    x = Conv2D(filters=96, kernel_size=1, activation='relu',strides=1, padding='same')(ms)
    return x

def attention_spatial(inputs2):

    a = Dense((inputs2.shape[3]).value, activation='softmax')(inputs2)
    return a

def attention_vertical(inputs):
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)
    
    a_probs = Permute((3,2,1))(a)
    return a_probs

def attention_horizontal(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    return b_probs

class DGA(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(DGA, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)

        X, A1,A2,A3 = x
        A = (A1 + A2 + A3)
        concatenate2 = A1 * A2 + A1 * A3 + A2 * A3

        min1 = tf.minimum(A1, A2)
        minv = tf.minimum(min1, A3)
        max1 = tf.maximum(A1, A2)
        maxv = tf.maximum(max1, A3)
        alpfa = 0.7
        bta = 0.2
        gama = 0.1
        assig = alpfa * maxv + bta * (A - maxv - minv) + gama * minv #weight assignment

        concatenate3 = K.concatenate([X, assig], axis=3)
        
        return [concatenate2 , concatenate3]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim1 = (input_shape[0][0],image_size,image_size, 1 * self.units)
        input_dim2 = (input_shape[0][0], image_size, image_size, 2 * self.units)
        return [input_dim1, input_dim2]


class RE_ope(Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(RE_ope, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)
        M1, M2 = x
        n = 1
        reward = n * M1
        punishment = tf.zeros_like(M1)
        M1 = tf.where(M1 > 0.2, x=reward, y=punishment)

        A = tf.multiply(M1, M2)

        return A

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 1 * self.units)
        return input_dim

def RE_module(xx):
    m1 = Activation('swish')(xx)
    m2 = Activation('tanh')(xx)
    RE = RE_ope(128)([m1, m2])
    x = BatchNormalization()(RE)
    return x

def demo(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    print(CNNInput.shape)
    print("---")
    x = PFM(CNNInput)
    print(x.shape)
    x = AveragePooling2D(2, strides=1)(x)
    x1 = BatchNormalization()(x)

    xx = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(x)
    x = RE_module(xx)
    att_3 = attention_spatial(x)
    att_x2 = attention_vertical(x)
    att_x = attention_horizontal(x)
    G1, L2 = DGA(128)([x, att_x, att_x2, att_3])

    L2 = Reshape((81, 256))(L2)
    L2 = Conv1D(filters=384, kernel_size=3, strides=3, activation='relu')(L2)
    x = BatchNormalization()(x)
    L2 = Reshape((9, 9, 128))(L2)
    x = concatenate([G1, L2])
    x = AveragePooling2D(2, strides=1)(x)
    x2 = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
    x = BatchNormalization()(x)
    att_3 = attention_spatial(x)
    att_x2 = attention_vertical(x)
    att_x = attention_horizontal(x)
    G1, L2 = DGA(128)([x, att_x, att_x2, att_3])

    L2 = Reshape((64, 256))(L2)
    L2 = Conv1D(filters=512, kernel_size=3, strides=4, activation='relu')(L2)
    x = BatchNormalization()(x)
    print(L2)
    L2 = Reshape((8, 8, 128))(L2)
    x = concatenate([G1, L2])
    x = AveragePooling2D(2, strides=1)(x)
    x3 = BatchNormalization()(x)

    x1 = AveragePooling2D(2)(x1)
    x2 = AveragePooling2D(2)(x2)
    x3 = AveragePooling2D(4, strides=1)(x3)

    x = concatenate([x1, x2, x3])
    x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)  # #

    x = Dense(256, activation='relu')(x)
    F = Dropout(0.5)(x)

    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=[F])
    print(model.summary())
    cal_flops(model)
    return model

def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='aspn', lr=0.01):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'demo':
        model = demo(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'demo_':
        model = demo_(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use demo1 model')
        model = demo1(img_rows, img_cols, num_PC, nb_classes)

    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

def demo_(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    F = Reshape([img_rows * img_cols, num_PC])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=F)
    print(model(inputs=[CNNInput]))
    print(model.summary())
    cal_flops(model)

    return model

def demo1(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    #模型架构区
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(
        CNNInput)
    model = Model(inputs=[CNNInput], outputs=F)
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])

def get_callbacks(decay=0.0001):
    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * decay)

    callbacks = []
    callbacks.append(LearningRateScheduler(step_decay))

    return callbacks
