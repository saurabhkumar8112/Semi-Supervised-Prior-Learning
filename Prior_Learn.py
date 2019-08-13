from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib
matplotlib.use('Agg')
from keras import backend as K
from keras.engine import *
from keras.legacy import interfaces
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers import Dense, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Embedding
import tensorflow as tf
from random import shuffle

class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 
        
class _ConvSN(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        super(_ConvSN, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.spectral_normalization = spectral_normalization
        self.u = None
        
    def _l2normalize(self, v, eps=1e-12):
        return v / (K.sum(v ** 2) ** 0.5 + eps)
    
    def power_iteration(self, u, W):
        '''
        Accroding the paper, we only need to do power iteration one time.
        '''
        v = self._l2normalize(K.dot(u, K.transpose(W)))
        u = self._l2normalize(K.dot(v, W))
        return u, v
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        #Spectral Normalization
        if self.spectral_normalization:
            self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                     initializer=initializers.RandomNormal(0, 1),
                                     name='sn',
                                     trainable=False)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        
        if self.spectral_normalization:
            W_shape = self.kernel.shape.as_list()
            #Flatten the Tensor
            W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
            _u, _v = power_iteration(W_reshaped, self.u)
            #Calculate Sigma
            sigma=K.dot(_v, W_reshaped)
            sigma=K.dot(sigma, K.transpose(_u))
            #normalize it
            W_bar = W_reshaped / sigma
            #reshape weight tensor
            if training in {0, False}:
                W_bar = K.reshape(W_bar, W_shape)
            else:
                with tf.control_dependencies([self.u.assign(_u)]):
                    W_bar = K.reshape(W_bar, W_shape)

            #update weitht
            self.kernel = W_bar
        
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import time
import math
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import  Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import _Merge
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from functools import partial

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

BATCH_SIZE = 64
GRADIENT_PENALTY_WEIGHT = 10

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

# Custom Functions for Reparametrization

def fwd_sigmoid(x):
    return 1/(1+math.exp(-x))

def inv_sigmoid(s):
    return math.log(s/(1-s))

def fwd_softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

class Alpha_Layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Alpha_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.alpha = self.add_weight(name='alpha',shape=(self.output_dim,),initializer=keras.initializers.Constant(value=[1]*self.output_dim),trainable=True)
        super(Alpha_Layer, self).build(input_shape)

    def H_function(self,a_i,alpha_batch):
        alpha_subtract=K.tf.subtract(alpha_batch,a_i)

        alpha_func=K.tf.keras.backend.hard_sigmoid(10000000*K.tf.pow(alpha_subtract,3))

        return K.tf.subtract(float(1),alpha_func)

    def call(self,x):
        alpha_batch = x
        ldim= self.output_dim
        Wj = [None]*ldim
        Hmj = [None]*ldim

        alpha_softmax = K.tf.nn.softmax(self.alpha)

        alpha_cml = K.tf.cumsum(alpha_softmax)

        for j in range(0,ldim):
            Hmj[j] = self.H_function(alpha_cml[j:(j+1)],alpha_batch)


        Wj[0] = Hmj[0]

        for j in range(1,ldim):
            Wj[j] = (1-Hmj[j-1])*(Hmj[j])

        onehot_concat = K.tf.concat(Wj,axis=-1)

        return onehot_concat

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def disc_loss(y):
    y_true=y[0]
    y_pred=y[1]
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def alpha_loss_dec(y):
    y_alpha=y[1]
    y_dec=y[0]
    
    sum_dec = K.tf.reduce_mean(y_dec,axis=0)
    sum_alpha = K.tf.reduce_mean(y_alpha,axis=0)
    
    return K.sum(sum_dec * K.log(sum_dec / sum_alpha))


def identity_loss(y_true, y_pred):
    return y_pred

def categorical_crossentropy_sp(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def mean_squared_error_sp(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def binary_crossentropy_sp(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def special_crossentropy_loss(x):
    true_val=K.stop_gradient(x[0])
    pred_val=x[1]
    return binary_crossentropy_sp(true_val,pred_val)


def special_mse_loss(x):
    true_val=K.stop_gradient(x[0])
    pred_val=x[1]
    return mean_squared_error_sp(true_val,pred_val)
    
        
#%%        


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])



def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1,z_dim = 50, batch_size = 2, alpha_dim=2):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.Dec = None   # decoder
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.AE = None  # Autoencoder model
        self.DE = None  # Decoder Model
        self.AlphaM = None # Alpha model
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.alpha_dim = alpha_dim

    def noise_add(self,tup):
        noise  = tup[0]
        alpha = tup[1]
        alpha_concat_zero = K.tf.zeros([self.batch_size,self.z_dim-self.alpha_dim])
        alpha_full = K.tf.concat([alpha_concat_zero,alpha],axis = -1)
        output = K.tf.add(alpha_full,noise)
        return output

    def decoder(self):
        if self.Dec:
            return self.Dec

        z_dim = self.z_dim
        dropout = 0.4
        depth = 32
        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x = Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*4, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*8, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*16, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Flatten()(x)
        z = Dense(z_dim)(x)
        z = LeakyReLU(alpha=0.2)(z)

        x1 =  Dense(self.alpha_dim)(z)
        x1 =  Activation('softmax')(x1)

        self.Dec = Model(img_input, [z, x1], name='Decoder')

        return self.Dec

    def discriminator(self):
        if self.D:
            return self.D

        depth = 64
        dropout = 0.4

        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x =  ConvSN2D(depth*1, kernel_size=5, strides=2,kernel_initializer='glorot_uniform', padding='same')(img_input)
        #x =  Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  ConvSN2D(depth*2, kernel_size=5, strides=2,kernel_initializer='glorot_uniform', padding='same')(x)
        #x =  Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  ConvSN2D(depth*4, kernel_size=5, strides=2,kernel_initializer='glorot_uniform', padding='same')(x)
        #x =  Conv2D(depth*4, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  ConvSN2D(depth*8, kernel_size=5, strides=2,kernel_initializer='glorot_uniform', padding='same')(x)
        #x =  Conv2D(depth*8, 5, strides=1, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Flatten()(x)
        x =  DenseSN(1,kernel_initializer='glorot_uniform')(x)
        #x =  Dense(1)(x)

        self.D = Model(img_input, x, name='Discriminator')

        return self.D

    def generator(self):
        if self.G:
            return self.G

        z_dim = self.z_dim

        dropout = 0.3
        depth = 32+32+32+32
        dim = 7

        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        alpha_l=Alpha_Layer(self.alpha_dim)
        alpha_l.name="Alpha"
        alpha=alpha_l(ip_alpha)
        noise = Lambda(self.noise_add)([ip_noise,alpha])


        x =  Dense(1024)(noise)
        x =  BatchNormalization(momentum=0.9)(x)

        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dense(dim*dim*depth)(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Reshape((dim, dim, depth))(x)
        x =  Dropout(dropout)(x)

        x =  Conv2DTranspose(int(depth/2), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  UpSampling2D()(x)
        x =  Conv2DTranspose(int(depth/4), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(int(depth/8),5 , padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(int(depth/16), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  UpSampling2D()(x)
        x =  Conv2DTranspose(int(depth/32),5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(1, 5, padding='same')(x)
        x =  Activation('sigmoid')(x)
        self.G = Model([ip_alpha,ip_noise],[x,alpha,noise], name='Generator')

        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM

        z_dim = self.z_dim

        for layer in self.generator().layers:
            layer.trainable=False
        self.generator().trainable=False

        for layer in self.discriminator().layers:
            layer.trainable=True
        self.discriminator().trainable=True



        real_samples = Input(shape=(self.img_rows, self.img_cols, self.channel))
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        generated_samples_for_discriminator = self.generator()([ip_alpha,ip_noise])[0]
        discriminator_output_from_generator = self.discriminator()(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = self.discriminator()(real_samples)


        averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])

        averaged_samples_out = self.discriminator()(averaged_samples)


        partial_gp_loss = partial(gradient_penalty_loss,averaged_samples=averaged_samples, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)


        discriminator_model = Model(inputs=[real_samples, ip_alpha,ip_noise],outputs=[discriminator_output_from_real_samples,discriminator_output_from_generator,averaged_samples_out])


        self.DM = discriminator_model
        self.DM.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM

        z_dim = self.z_dim

        for layer in self.generator().layers:
            layer.trainable=True
        self.generator().trainable=True
        (self.generator().get_layer('Alpha')).trainable=False

        for layer in self.discriminator().layers:
            layer.trainable=False
        self.discriminator().trainable=False

        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))

        H = self.generator()([ip_alpha,ip_noise])
        V = self.discriminator()(H[0])
        self.AM = Model([ip_alpha,ip_noise], V)
        self.AM.compile(loss=wasserstein_loss, optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), metrics=['acc'])
        return self.AM


    def autoencoder_model(self):
        if self.AE:
            return self.AE

        z_dim = self.z_dim

        optimizer = Adam(lr=0.0001, decay=6e-8)
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        self.generator().trainable=True
        for layer in self.generator().layers:
            layer.trainable=True
        (self.generator().get_layer('Alpha')).trainable=False
        H = self.generator()([ip_alpha,ip_noise])
        [Vz, Vx] = self.decoder()(H[0])

        Vz_l = Lambda(special_mse_loss)([H[2],Vz])
        Vx_l = Lambda(special_crossentropy_loss)([H[1],Vx])


        self.AE = Model([ip_alpha,ip_noise], [Vz_l, Vx_l])
        self.AE.compile(loss=[identity_loss, identity_loss],loss_weights=[10.0, 1.0], optimizer=optimizer)
        return self.AE

    def alpha_model(self):
        if self.AlphaM:
            return self.AlphaM

        z_dim = self.z_dim

        optimizer = Adam(lr=0.001, decay=3e-4)
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))

        Real_Imgs = Input(shape=(self.img_rows, self.img_cols, self.channel))

        self.generator().trainable=True
        for layer in self.generator().layers:
            layer.trainable=False
        (self.generator().get_layer('Alpha')).trainable=True
        H = self.generator()([ip_alpha,ip_noise])
        self.discriminator().trainable=False
        for layer in self.discriminator().layers:
            layer.trainable=False

        self.decoder().trainable=False
        for layer in self.decoder().layers:
            layer.trainable=False

        alpha = H[1]

        _ , dec_op = self.decoder()([Real_Imgs])

        V_dec = Lambda(alpha_loss_dec)([dec_op,alpha])

        V = V_dec

        self.AlphaM = Model([ip_alpha,ip_noise,Real_Imgs], V)
        self.AlphaM.compile(loss=identity_loss, optimizer=optimizer)
        return self.AlphaM

    def decoder_model(self):
        if self.DE:
            return self.DE
        optimizer = Adam(lr=0.0001, decay=6e-8)
        self.decoder().trainable=True
        for layer in self.decoder().layers:
            layer.trainable=True
        de_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        [Vz, Vx] = self.decoder()(de_input)
        self.DE = Model(de_input, [Vz, Vx])
        self.DE.compile(loss=['mse', keras.losses.categorical_crossentropy], loss_weights=[0, 1.0],  optimizer=optimizer, metrics=['acc'])
        return self.DE

class MNIST_NEMGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.z_dim = 20
        self.alpha_dim = 2
        self.batch_size = BATCH_SIZE

        self.x_train = input_data.read_data_sets(os.path.expanduser("~/prasdisk2/sngan_prior/SpectralNormalizationKeras/MNIST_data"), one_hot=True).train.images
        self.ylabel = input_data.read_data_sets(os.path.expanduser("~/prasdisk2/sngan_prior/SpectralNormalizationKeras/MNIST_data"), one_hot=True).train.labels
        self.x_test = input_data.read_data_sets(os.path.expanduser("~/prasdisk2/sngan_prior/SpectralNormalizationKeras/MNIST_data"), one_hot=True).test.images
        self.x_test=self.x_test.reshape( (self.x_test.shape[0],28,28,1 ) )
        self.ylabel_test = input_data.read_data_sets(os.path.expanduser("~/prasdisk2/sngan_prior/SpectralNormalizationKeras/MNIST_data"), one_hot=True).test.labels
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)

        temp=np.argmax(self.ylabel, axis=1)
        indx_3=[i for i, x in enumerate(temp) if x == 3]
        indx_3=np.random.choice(indx_3, size=5500, replace=False, p=None)
        indx_5=[i for i, x in enumerate(temp) if x == 5]
        indx_5=np.random.choice(indx_5, size=1375, replace=False, p=None)

        indx = list(indx_5)+list(indx_3)
        shuffle(indx)

        self.x_train = self.x_train[indx]
        self.ylabel = self.ylabel[indx]

        temp_test=np.argmax(self.ylabel_test, axis=1)
        indx_3_test=[i for i, x in enumerate(temp_test) if x == 3]
        indx_3_test=np.random.choice(indx_3_test, size=1000, replace=False, p=None)
        indx_5_test=[i for i, x in enumerate(temp_test) if x == 5]
        indx_5_test=np.random.choice(indx_5_test, size=250, replace=False, p=None)

        indx_test = list(indx_5_test)+list(indx_3_test)
        shuffle(indx_test)

        self.x_test = self.x_test[indx_test]
        self.ylabel_test = self.ylabel_test[indx_test]

        class_0 = [3]
        class_1 = [5]

        class_x = [class_0,class_1]

        x_cat = []
        y_cat = []
        for j in range(0,2):
            ind_list = [i for i in range(0, self.ylabel.shape[0]) if np.argmax(self.ylabel[i]) in class_x[j]]

            x_app = self.x_train[ind_list]

            y_app = np.zeros((x_app.shape[0],2))

            y_app[:,j]=1

            x_cat.append(x_app)
            y_cat.append(y_app)


        self.x_train = np.concatenate(x_cat)
        self.ylabel = np.concatenate(y_cat)


        x_cat = []
        y_cat = []
        for j in range(0,2):
            ind_list = [i for i in range(0, self.ylabel_test.shape[0]) if np.argmax(self.ylabel_test[i]) in class_x[j]]

            x_app = self.x_test[ind_list]

            y_app = np.zeros((x_app.shape[0],2))

            y_app[:,j]=1

            x_cat.append(x_app)
            y_cat.append(y_app)


        self.x_test = np.concatenate(x_cat)
        self.ylabel_test = np.concatenate(y_cat)

        self.x_test,self.ylabel_test = self.balancer(self.x_test,self.ylabel_test,2)

        self.DCGAN = DCGAN(alpha_dim=self.alpha_dim,batch_size=self.batch_size,z_dim=self.z_dim)
        print('D_model Summary:')
        self.discriminator =  self.DCGAN.discriminator_model()
        self.discriminator.summary()
        print('G_model Summary:')
        self.generator = self.DCGAN.generator()
        self.generator.summary()
        print('DE_model Summary:')
        self.decoder = self.DCGAN.decoder()
        self.decoder.summary()
        print('AD_model Summary:')
        self.adversarial = self.DCGAN.adversarial_model()
        self.adversarial.summary()
        print('AE_model Summary:')
        self.autoencoder = self.DCGAN.autoencoder_model()
        self.autoencoder.summary()
        print('Alpha_model Summary:')
        self.alphamodel = self.DCGAN.alpha_model()
        self.alphamodel.summary()
        print('Decoder_model Summary:')
        self.decoder_model = self.DCGAN.decoder_model()
        self.decoder_model.summary()


    def fixed_dist_gen(self,batch_size=256,minv=0,maxv=1,samples=10000000):
        out = np.linspace(minv, maxv, num=samples)
        out=out.reshape((samples,1))
        while(1):
           out_batch=out[np.random.randint(0,samples,batch_size)]

           yield out_batch

    def fixed_noise_gen(self,batch_size=256):
        return np.random.uniform(-1*0.3,0.3, (batch_size,self.z_dim) )


    def noise_gen_plot(self,alpha_softmax,batch_size=100,class_i=0):

        acc_alpha = np.cumsum(alpha_softmax)

        acc_alpha = [0] + list(acc_alpha)

        minv = acc_alpha[class_i]
        maxv = acc_alpha[class_i+1]
        optim_alpha = np.linspace(minv, maxv, num=batch_size)

        return optim_alpha, np.random.uniform(-1*0.3,0.3, (batch_size,self.z_dim) )

    def plot_images_save(self,alpha_batch,noise_batch,path=None):
        images = self.generator.predict([alpha_batch,noise_batch],batch_size = self.batch_size)[0]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(10, 10, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')

        plt.subplots_adjust(wspace=0.0000000001, hspace=0.0000000001, left=0, right=1, bottom=0, top=1)
        plt.savefig(path)
        plt.close('all')


    def balancer(self,x,y,n_classes):
        num_samples = x.shape[0]

        count_l = [0 for i in range(0,n_classes)]

        for i in range(0,num_samples):
            count_l[np.argmax(y[i])]+=1

        max_count = max(count_l)

        x_concat = []
        y_concat = []

        for i in range(0,n_classes):

            if(max_count-count_l[i]>0):

                defi = max_count-count_l[i]

                x_ind = [k for k in range(0, num_samples) if np.argmax(y[k])==i]

                x_add = x[x_ind][np.random.randint(0, len(x_ind), size=defi)]

                y_add = np.zeros((defi,n_classes))

                y_add[:,i] = 1

                x_concat.append(x_add)
                y_concat.append(y_add)

        x_concat.append(x)
        y_concat.append(y)

        return np.concatenate(x_concat),np.concatenate(y_concat)

    def compute_purity(self,y_pred, y_true):
        clusters = set(y_pred)
        correct = 0
        for cluster in clusters:

            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)

        return float(correct) / len(y_pred)


    def calc_metrics(self):

        xtest,ytest=(self.x_test,self.ylabel_test)

        y_pred_z, y_pred_y = self.decoder.predict(xtest)


        km = KMeans(n_clusters=max(self.alpha_dim, len(np.unique(ytest.argmax(axis=-1)))), random_state=0).fit(y_pred_z)

        labels_pred = km.labels_

        purity = self.compute_purity(labels_pred, ytest.argmax(axis=-1))
        ari = adjusted_rand_score(ytest.argmax(axis=-1), labels_pred)
        nmi = normalized_mutual_info_score(ytest.argmax(axis=-1), labels_pred)

        print('Purity: ',purity)
        print("NMI: ",nmi)
        print('ARI: ',ari)


    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        fig_path = './figure/'
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        model_path = './model/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        gen_n=self.fixed_dist_gen(batch_size=batch_size)

        n_sup = 50
        n_classes = 2

        ind_list = [np.random.randint(0, self.x_train.shape[0], size=n_sup)]

        retrain_data = self.x_train[ind_list]
        retrain_labels = self.ylabel[ind_list]

        ret_z = np.zeros((self.batch_size,self.z_dim))

        retrain_data, retrain_labels = self.balancer(retrain_data, retrain_labels,n_classes)

        ret_size = retrain_data.shape[0]

        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]

            alpha_batch= next(gen_n)

            noise_batch=self.fixed_noise_gen(batch_size=batch_size)

            positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

            d_loss = self.discriminator.train_on_batch([images_train,alpha_batch,noise_batch],[positive_y, negative_y, dummy_y])

            a_loss = self.adversarial.train_on_batch([alpha_batch,noise_batch], positive_y)


            ae_loss = self.autoencoder.train_on_batch([alpha_batch,noise_batch], [noise_batch, noise_batch])

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            log_mesg = "%s  [AE loss total: %f, AE loss L1: %f, AE loss bce: %f]" % (log_mesg, ae_loss[0], ae_loss[1], ae_loss[2])


            alpha_loss = -1
            if( (i+1)>10000 and ((i+1)%1) == 0 ):
                alpha_loss = self.alphamodel.train_on_batch([alpha_batch,noise_batch,images_train], [np.zeros([batch_size, 1])])

            alpha_cur_val = self.alphamodel.get_layer('Generator').get_layer('Alpha').get_weights()[0]
            print('Alpha_loss %f, Alpha:'%(alpha_loss,),fwd_softmax(alpha_cur_val))


            if(i+1>100 and (i+1)%10==0):
                 if((i+1)%100==0):
                     loo=100
                 else:
                     loo=2
                 print('Doing supervised training on decoder....... For:',loo)
                 for jj in range(0,loo):
                     rt_indices=np.random.randint(low=0,high=ret_size,size=batch_size)
                     de_loss = self.decoder_model.train_on_batch(retrain_data[rt_indices], [ret_z, retrain_labels[rt_indices]])

                 log_mesg_rtrain = "[DE loss total: %f, DE loss L2: %f, DE loss bce: %f, DE acc_z: %f, DE acc_l: %f]" % ( de_loss[0], de_loss[1], de_loss[2], de_loss[3], de_loss[4])
                 print(log_mesg_rtrain)


            if (i+1)%save_interval==0:
                self.calc_metrics()
                for j in range(0,self.alpha_dim):
                    alpha_plot,noise_plot=self.noise_gen_plot(fwd_softmax(alpha_cur_val),batch_size=batch_size,class_i=j)
                    self.plot_images_save(alpha_plot,noise_plot,path='./figure/epoch_%d_mode_%d_.png' % (i+1,j))

            print(log_mesg)

            if (i+1)%5000==0:
                print('*********************Saving Weights***********************')
                self.discriminator.save_weights(os.path.expanduser('./model/gan_dircriminator_epoch_%d.h5' % (i+1)))
                self.generator.save_weights(os.path.expanduser('./model/gan_generator_epoch_%d.h5' % (i+1)))
                self.adversarial.save_weights(os.path.expanduser('./model/gan_adversarial_epoch_%d.h5' % (i+1)))
                self.decoder.save_weights(os.path.expanduser('./model/gan_decoder_epoch_%d.h5' % (i+1)))
                self.autoencoder.save_weights(os.path.expanduser('./model/gan_autoencoder_epoch_%d.h5' % (i+1)))


if __name__ == '__main__':
    mnist_nemgan = MNIST_NEMGAN()
    timer = ElapsedTimer()
    mnist_nemgan.train(train_steps=200000, batch_size=BATCH_SIZE, save_interval=1000)
    timer.elapsed_time()
