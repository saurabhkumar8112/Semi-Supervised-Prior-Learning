#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 04:07:06 2019

@author: aravind
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import time
import random
import math
#from keras.datasets import mnist
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape, Lambda, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Dense, Activation,BatchNormalization
from keras.layers import activations, initializers, regularizers, constraints
from pdb import set_trace as trace
import gc



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

#%%
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
        
        alpha_func=K.tf.keras.backend.hard_sigmoid(100000000000*K.tf.pow(alpha_subtract,3))
        
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
    
#def alpha_loss(y_true, y_pred):
#    return K.tf.divide(1, K.tf.add( K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1),1))
        
def disc_loss(y):
    y_true=y[0]
    y_pred=y[1]
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

# def alpha_loss(y):
#     y_real=y[0]
#     y_fake=y[1]
    
#     y_tot = K.tf.add(y_real,y_fake)
    
#     y_tot = K.tf.add(y_tot,1)
    
#     return K.tf.divide(1,y_tot)

def alpha_loss(y):
    y_real=y[0]
    y_fake=y[1]
    
    y_tot = K.tf.add(y_real,y_fake)

    return K.tf.multiply(-1.0,y_tot)


def alpha_loss_dec(y):
    y_alpha=y[1]
    y_dec=y[0]
    
    sum_dec = K.tf.reduce_mean(y_dec,axis=0)
    sum_alpha = K.tf.reduce_mean(y_alpha,axis=0)
    
    return K.tf.reduce_mean(K.tf.square(sum_alpha - sum_dec))

def add_lambda(y):
    y_disc=y[0]
    y_dec=y[1]
    return 0*y_disc+1*y_dec


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

def amsoftmax_loss(y):
    y_true=y[0]
    y_pred=y[1]
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss
    

def amsoftmax_loss_dec(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss


        
#%%        


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1,alpha_dim=2,batch_size=2,z_dim=10):
        
        self.alpha_dim=alpha_dim
        
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.Dec = None   # decoder
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.AE = None  # Autoencoder model
        self.DE = None
        self.AlphaM = None # Alpha model
        self.batch_size = batch_size
        self.z_dim = z_dim
        
    def noise_add(self,tup):
        noise  = tup[0]
        alpha = tup[1]
        alpha_concat_zero = K.tf.zeros([self.batch_size,self.z_dim-self.alpha_dim])
        alpha_full = K.tf.concat([alpha_concat_zero,alpha],axis = -1)
        output = K.tf.add(alpha_full,noise)
        return output
        
    def decoder(self, z_dim = 10):
        if self.Dec:
            return self.Dec
        # self.Dec = Sequential()   
        dropout = 0.4
        depth = 16
        # input_shape = (self.img_rows, self.img_cols, self.channel)
        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x = Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        #x = Conv2D(depth*4, 5, strides=2, padding='same')(x)
        #x = LeakyReLU(alpha=0.2)(x)
        #x = Dropout(dropout)(x)

        x = Conv2D(depth*8, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(depth*16, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        # x = Conv2D(depth*32, 5, strides=1, padding='same')(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # x = Dropout(dropout)(x)
        
        x = Flatten()(x)
        z = Dense(z_dim)(x)
        z = LeakyReLU(alpha=0.2)(z)
#         self.Dec.add(Dense(10))
#         self.Dec.add(Activation('softmax'))
        #x1 =  AMSoftmax(2, 30, 0.4)(z)
        
        x1 =  Dense(self.alpha_dim)(z)
        
        #x1 =  Activation('softmax')(x1)
        x1 =  AMSoftmax(self.alpha_dim, 30, 0.4)(x1)
        # self.Dec.add(output)
#        self.Dec.add(Dense(units=1,activation='sigmoid'))
        self.Dec = Model(img_input, [z, x1], name='Decoder')
        # self.Dec.summary()
        return self.Dec

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        # self.D = Sequential()
        depth = 64
        #depth = 128
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        # input_shape = (self.img_rows, self.img_cols, self.channel)
        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x =  Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*4, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*8, 5, strides=1, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        # Out: 1-dim probability
        x =  Flatten()(x)
        x =  Dense(1)(x)
        x =  Activation('sigmoid')(x)
        self.D = Model(img_input, x, name='Discriminator')
        # self.D.summary()
        return self.D

    def generator(self, z_dim = 10):
        if self.G:
            return self.G
        # self.G = Sequential()
        dropout = 0.3
        depth = 32+32+32+32
        #depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        alpha_l=Alpha_Layer(self.alpha_dim)
        alpha_l.name="Alpha"
        alpha=alpha_l(ip_alpha)
        #noise = Concatenate()([ip_noise,alpha])
        noise = Lambda(self.noise_add)([ip_noise,alpha])
        
        x =  Dense(100)(noise)
        x =  BatchNormalization(momentum=0.9)(x)
        # x =  Dense(100)(x)
        # x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dense(dim*dim*depth)(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Reshape((dim, dim, depth))(x)
        x =  Dropout(dropout)(x)

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        x =  UpSampling2D()(x)
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
        
        x =  Conv2DTranspose(int(depth/32),5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(1, 5, padding='same')(x)
        x =  Activation('sigmoid')(x)
        self.G = Model([ip_alpha,ip_noise], [x,alpha,noise], name='Generator')
        # self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0002, decay=6e-8)
        # self.DM = Sequential()
        self.discriminator().trainable=True
        for layer in self.discriminator().layers:
            layer.trainable=True
        self.DM = self.discriminator()
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        return self.DM

    def adversarial_model(self, z_dim = 10):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0001, decay=3e-8)
        # self.AM = Sequential()
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        self.generator().trainable=True
        for layer in self.generator().layers:
            layer.trainable=True
        (self.generator().get_layer('Alpha')).trainable=False
        H = self.generator()([ip_alpha,ip_noise])
        self.discriminator().trainable=False
        for layer in self.discriminator().layers:
            layer.trainable=False
        V = self.discriminator()(H[0])
        # self.AM.add(self.discriminator())
        self.AM = Model([ip_alpha,ip_noise], V)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        return self.AM
    
    def alpha_model(self, z_dim = 10):
        if self.AlphaM:
            return self.AlphaM
        optimizer = Adam(lr=0.001, decay=3e-4)
        # self.AM = Sequential()
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        
        op_real = Input(shape=(1,))
        op_fake = Input(shape=(1,))
        
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
        
        V_fake = self.discriminator()(H[0])
        
        V_real = self.discriminator()(Real_Imgs)
        
        L_real = Lambda(disc_loss)([op_real,V_real])
        
        L_fake = Lambda(disc_loss)([op_fake,V_fake])
        
        V_disc = Lambda(alpha_loss)([L_real,L_fake])
        
        V = Lambda(add_lambda)([V_disc,V_dec])
        
        # self.AM.add(self.discriminator())
        self.AlphaM = Model([ip_alpha,ip_noise,Real_Imgs,op_real,op_fake], V)
        self.AlphaM.compile(loss=identity_loss, optimizer=optimizer)
        return self.AlphaM
    
    def autoencoder_model(self, z_dim = 10):
        if self.AE:
            return self.AE
        optimizer = Adam(lr=0.0001, decay=6e-8)
        # self.AE = Sequential()
        ip_alpha=Input(shape=(1,))
        ip_noise=Input(shape=(z_dim,))
        self.generator().trainable=True
        for layer in self.generator().layers:
            layer.trainable=True
        (self.generator().get_layer('Alpha')).trainable=False
        H = self.generator()([ip_alpha,ip_noise])
        # self.AE.add(self.generator())
        # self.AE.add(self.decoder())
        [Vz, Vx] = self.decoder()(H[0])
        
        Vz_l = Lambda(special_mse_loss)([H[2],Vz])
        #Vx_l = Lambda(special_crossentropy_loss)([H[1],Vx])
        Vx_l = Lambda(amsoftmax_loss)([H[1],Vx])
        
        self.AE = Model([ip_alpha,ip_noise], [Vz_l, Vx_l])
        #self.AE.compile(loss=[identity_loss, identity_loss], optimizer=optimizer, metrics=['acc'])
        self.AE.compile(loss=[identity_loss, identity_loss],loss_weights=[10.0, 1.0], optimizer=optimizer)
        return self.AE
    
    def decoder_model(self):
        if self.DE:
            return self.DE
        optimizer = Adam(lr=0.0001, decay=6e-8)
        # self.AE = Sequential()
        self.decoder().trainable=True
        for layer in self.decoder().layers:
            layer.trainable=True
        de_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        # self.AE.add(self.generator())
        # self.AE.add(self.decoder())
        [Vz, Vx] = self.decoder()(de_input)
        self.DE = Model(de_input, [Vz, Vx])
        self.DE.compile(loss=['mse', amsoftmax_loss_dec], loss_weights=[0, 1.0],  optimizer=optimizer, metrics=['acc'])
        return self.DE

#%%
class AMSoftmax(Layer):
    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.m

        e_costheta = K.exp(self.s * dis_cosin)
        e_psi = K.exp(self.s * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp
        return output

#%%

class MNIST_NEMGAN(object):
    def __init__(self,alpha_dim=2,batch_size=2,z_dim=10):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.z_dim = z_dim
        self.alpha_dim = alpha_dim
        self.batch_size = batch_size
        
        self.x_train = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).train.images
        self.ylabel = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).train.labels
        self.x_test = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).test.images
        self.x_test=self.x_test.reshape( (self.x_test.shape[0],28,28,1 ) )
        self.ylabel_test = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).test.labels
        
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        print(self.ylabel.shape)
        print(self.x_train.shape)
        self.DCGAN = DCGAN(alpha_dim=self.alpha_dim,batch_size=self.batch_size,z_dim=self.z_dim)
#        print('D_model Summary:')
#        self.discriminator =  self.DCGAN.discriminator_model()
#        self.discriminator.summary()
        print('G_model Summary:')
        self.generator = self.DCGAN.generator(z_dim = self.z_dim)
        #self.generator.summary()
#        print('DE_model Summary:')
#        self.decoder = self.DCGAN.decoder(z_dim = self.z_dim)
#        self.decoder.summary()
#        print('AD_model Summary:')
#        self.adversarial = self.DCGAN.adversarial_model(z_dim = self.z_dim)
#        self.adversarial.summary()
#        print('AE_model Summary:')
#        self.autoencoder = self.DCGAN.autoencoder_model(z_dim = self.z_dim)
#        self.autoencoder.summary()
#        print('Alpha_model Summary:')
#        self.alphamodel = self.DCGAN.alpha_model(z_dim = self.z_dim)
#        self.alphamodel.summary()
#        print('Decoder_model Summary:')
#        self.decoder_model = self.DCGAN.decoder_model()
#        self.decoder_model.summary()

    def data_gen_test(self):
        print(self.x_test.shape)
        fourss=np.zeros((1,self.x_test.shape[1],self.x_test.shape[2],self.x_test.shape[3]))
        zeross=np.zeros((1,self.x_test.shape[1],self.x_test.shape[2],self.x_test.shape[3]))
        
        print('Making Test Dataset....')
        for i in range(0,self.x_test.shape[0]):
            if(np.argmax(self.ylabel_test[i])==3):
                fourss=np.vstack([fourss,self.x_test[i].reshape(1,self.x_test.shape[1],self.x_test.shape[2],self.x_test.shape[3])])
            if(np.argmax(self.ylabel_test[i])==5):
                zeross=np.vstack([zeross,self.x_test[i].reshape(1,self.x_test.shape[1],self.x_test.shape[2],self.x_test.shape[3])])
        
        print('Done Making Test Dataset')
        zeross=zeross[1:]
        fourss=fourss[1:]
        
        one_len=zeross.shape[0]
        two_len=fourss.shape[0]
        
        y1=np.zeros( (one_len,2) )
        y1[:,0]=1
        y2=np.zeros( (two_len,2) )
        y2[:,1]=1
        y_batch=np.concatenate(   (y1,y2)  )
        
        x_batch= np.concatenate((zeross,fourss))
        
        return x_batch,y_batch


    def data_gen_onefour_UB(self,p1,p2,d1,d2,batch_size=256):
        #print(self.x_train.shape)
        fourss=np.zeros((1,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3]))
        zeross=np.zeros((1,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3]))
        
        print('Making Dataset....')
        for i in range(0,self.x_train.shape[0]):
            if(np.argmax(self.ylabel[i])==d1):
                fourss=np.vstack([fourss,self.x_train[i].reshape(1,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3])])
            if(np.argmax(self.ylabel[i])==d2):
                zeross=np.vstack([zeross,self.x_train[i].reshape(1,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3])])
        
        print('Done Making Dataset')
        
        tot_cons = 4100
        if(p1>p2):
            fourss=fourss[1:int(tot_cons)]
            zeross=zeross[1:int(tot_cons*(p2/p1))]
        else:
            fourss=fourss[1:int(tot_cons*(p1/p2))]
            zeross=zeross[1:int(tot_cons)]
        dataset = np.concatenate([zeross,fourss])

        ind_list=list(range(0,dataset.shape[0]))
        random.shuffle(ind_list)
        
        return dataset
    
#        while(1):
#            x_batch=dataset[ind_list[:batch_size]]
#            ind_list=ind_list[batch_size:]
#            if(len(ind_list)<batch_size):
#                ind_list=list(range(0,dataset.shape[0]))
#                random.shuffle(ind_list)
#                    
#        
#            yield x_batch,None,dataset.shape

                

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
            
            
        #plt.tight_layout()
        plt.subplots_adjust(wspace=0.0000000001, hspace=0.0000000001, left=0, right=1, bottom=0, top=1)
        plt.savefig(path)
        plt.close('all')
    

   
    def train(self, train_steps=2000, save_interval=1000):
        batch_size = self.batch_size
        gen=self.data_gen_onefour_UB(0.2,0.8,batch_size=batch_size)
        xtest,ytest=self.data_gen_test()
        gen_n=self.fixed_dist_gen(batch_size=batch_size)
        
        if not os.path.exists('./figure'):
            os.mkdir('./figure')
        if not os.path.exists('./model'):
            os.mkdir('./model')
        
        
        n_sup=200
        
        retrain_3=self.x_train[ [i for i in range(0,self.x_train.shape[0]) if np.argmax(self.ylabel[i])==3][:n_sup] ]
        retrain_5=self.x_train[ [i for i in range(0,self.x_train.shape[0]) if np.argmax(self.ylabel[i])==5][:n_sup] ]
        
        #trace()
        
        retrain_data=np.concatenate([retrain_3,retrain_5])
        retrain_labels=np.zeros((2*n_sup,2))
        retrain_labels[:n_sup,0]=1
        retrain_labels[n_sup:,1]=1
        
        ret_nohts=np.zeros((batch_size,self.z_dim))
        
        label_test = True
        for i in range(train_steps):
            images_train,_,_=next(gen)
            
            alpha_batch= next(gen_n)
            
            noise_batch=self.fixed_noise_gen(batch_size=batch_size)
            
            images_fake = self.generator.predict([alpha_batch,noise_batch],batch_size = batch_size)[0]

            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
        
            a_loss = self.adversarial.train_on_batch([alpha_batch,noise_batch], y)
            
            ae_loss = self.autoencoder.train_on_batch([alpha_batch,noise_batch], [noise_batch, noise_batch])
            
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i+1, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            #log_mesg = "%s  [AE loss total: %f, AE loss L2: %f, AE loss bce: %f, AE acc_z: %f,%f, AE acc_l: %f,%f]" % (log_mesg, ae_loss[0], ae_loss[1], ae_loss[2], ae_loss[3][0], ae_loss[4][0], ae_loss[3][1], ae_loss[4][1])
            log_mesg = "%s  [AE loss total: %f, AE loss L2: %f, AE loss bce: %f]" % (log_mesg, ae_loss[0], ae_loss[1], ae_loss[2])
            
            alpha_loss = -1
            if( (i+1)>20000 and ((i+1)%1) == 0 ):
            #if(True):
            #if(False):
                alpha_loss = self.alphamodel.train_on_batch([alpha_batch,noise_batch,images_train,np.ones([batch_size, 1]),np.zeros([batch_size, 1])], [np.zeros([batch_size, 1])])
                
            alpha_cur_val = self.alphamodel.get_layer('Generator').get_layer('Alpha').get_weights()[0]
            print('Alpha_loss %f, Alpha:'%(alpha_loss,),fwd_softmax(alpha_cur_val))
            
            
            if(i+1>10000 and (i+1)%10==0):
            #if(False):
                if(label_test):
                    label_test = False
                    _, pred_y_test=test_decoder.predict(retrain_3)
                    sum_pred = np.sum(pred_y_test,axis=0)
                    print('Calculated Sum For mode swap,:',sum_pred)
                    if(sum_pred[0]<sum_pred[1]):
                        retrain_labels=np.zeros((2*n_sup,2))
                        retrain_labels[:n_sup,1]=1
                        retrain_labels[n_sup:,0]=1


                if((i+1)%100==0):
                    loo=100
                else:
                    loo=2
                print('Doing supervised training on decoder....... For:',loo)
                for jj in range(0,loo):
                    rt_indices=np.random.randint(low=0,high=2*n_sup,size=batch_size)
                    de_loss = self.decoder_model.train_on_batch(retrain_data[rt_indices], [ret_nohts, retrain_labels[rt_indices]])
                
                log_mesg_rtrain = "[DE loss total: %f, DE loss L2: %f, DE loss bce: %f, DE acc_z: %f, DE acc_l: %f]" % ( de_loss[0], de_loss[1], de_loss[2], de_loss[3], de_loss[4])
                print(log_mesg_rtrain)
            
            if (i+1)%save_interval==0:
                for j in range(0,self.alpha_dim):
                    alpha_plot,noise_plot=self.noise_gen_plot(fwd_softmax(alpha_cur_val),batch_size=batch_size,class_i=j)
                    self.plot_images_save(alpha_plot,noise_plot,path='./figure/epoch_%d_mode_%d_.png' % (i+1,j))
                    
                    
            if((i+1)%100==0):
               test_decoder=self.autoencoder.layers[-3]
               
               pred_z, pred_y=test_decoder.predict(xtest)   
               
               count=0
               for j in range(0,pred_y.shape[0]):
                   if( np.argmax(pred_y[j])==np.argmax(ytest[j])  ):
                       count+=1.
                       
               print('Decoder_Accuracy = ',count/pred_y.shape[0])
            
            print(log_mesg)
            
            if (i+1)%5000==0:
                print('*********************Saving Weights***********************')
                self.discriminator.save_weights(os.path.expanduser('./model/gan_dircriminator_epoch_%d.h5' % (i+1)))
                self.generator.save_weights(os.path.expanduser('./model/gan_generator_epoch_%d.h5' % (i+1)))
                self.adversarial.save_weights(os.path.expanduser('./model/gan_adversarial_epoch_%d.h5' % (i+1)))
                self.decoder.save_weights(os.path.expanduser('./model/gan_decoder_epoch_%d.h5' % (i+1)))
                self.autoencoder.save_weights(os.path.expanduser('./model/gan_autoencoder_epoch_%d.h5' % (i+1)))

#%%

#batch_size = 16
#
#mnist_nemgan = MNIST_NEMGAN(alpha_dim=2,z_dim=10,batch_size=batch_size)
#
#mnist_nemgan.generator.load_weights('./4_0_ratio_var_gen_models/10.h5')
#
#full_classifier = keras.models.load_model('./cnns/cnn_mnist_10c.h5')
#
#req_layer = 'flatten_1'
##req_layer = 'final_logits'
#
#classifier = Model(inputs=full_classifier.input,outputs=full_classifier.get_layer(req_layer).output)
#
#real_data = mnist_nemgan.data_gen_onefour_UB(0.4,0.6,  4,0,  batch_size=None)
#
#random.shuffle(real_data)
#
#real_data = real_data[:(real_data.shape[0]//batch_size)*batch_size]
#
#noise_batch=mnist_nemgan.fixed_noise_gen(batch_size=real_data.shape[0])
#
#alpha_batch = next(mnist_nemgan.fixed_dist_gen(batch_size=real_data.shape[0]))
#
#gen_data = mnist_nemgan.generator.predict([alpha_batch,noise_batch],batch_size = batch_size)[0]
#
# #gen_data = real_data[:real_data.shape[0]//2]
# #real_data = real_data[real_data.shape[0]//2:]
#
#real_act = classifier.predict(real_data)
#
#gen_act = classifier.predict(gen_data)
##gen_act = classifier.predict(real_data)
#
#del mnist_nemgan
#del real_data
#del gen_data
#del full_classifier
#gc.collect()
#
#print('Calculating FCD......')
#fcd = tf.contrib.gan.eval.diagonal_only_frechet_classifier_distance_from_activations(tf.convert_to_tensor(real_act),tf.convert_to_tensor(gen_act) )
#
#sess = tf.Session()
#
#print('FCD: ',sess.run(fcd) )
#
#sess.close()

#%%

batch_size = 16

for i in range(5,10):

	print('**'*30)
	print('Model: ',i)
	mnist_nemgan = MNIST_NEMGAN(alpha_dim=2,z_dim=10,batch_size=batch_size)

	gen_name = './4_0_ratio_var_gen_models/' + str(10*i) +'.h5'

	print('Gen_file_name: ',gen_name)

	mnist_nemgan.generator.load_weights(gen_name)

	#mnist_nemgan.generator.layers[2].set_weights(np.asarray([[1,1]]))

	full_classifier = keras.models.load_model('./cnns/cnn_mnist_10c.h5')

	req_layer = 'flatten_1'
	#req_layer = 'final_logits'

	classifier = Model(inputs=full_classifier.input,outputs=full_classifier.get_layer(req_layer).output)

	real_data_fixed = mnist_nemgan.data_gen_onefour_UB(i/10,1-(i/10),  3,5,  batch_size=None)

	fcd_real_arr = []
	# for j in range(0,0):

	#     print('Real_data_iter: ',j)

	#     real_data = real_data_fixed.copy()

	#     np.random.shuffle(real_data)

	#     gen_data = real_data[:real_data.shape[0]//2]
	#     real_data = real_data[real_data.shape[0]//2:]

	#     real_data = real_data[:(real_data.shape[0]//batch_size)*batch_size]

	#     gen_data = gen_data[:(gen_data.shape[0]//batch_size)*batch_size]

	#     real_act = classifier.predict(real_data)

	#     gen_act = classifier.predict(gen_data)

	#     # max_val = max(np.max(gen_act),np.max(real_act))

	#     # gen_act = gen_act/max_val
	#     # real_act = real_act/max_val

	#     print('Calculating FCD......')
	#     fcd = tf.contrib.gan.eval.diagonal_only_frechet_classifier_distance_from_activations(tf.convert_to_tensor(real_act),tf.convert_to_tensor(gen_act) )

	#     sess = tf.Session()

	#     fcd_val = sess.run(fcd)

	#     print('FCD: ', fcd_val)

	#     fcd_real_arr.append(fcd_val)

	#     sess.close()
	# fcd_real_arr = np.asarray(fcd_real_arr)

	# print('Mean_real_fcd: ',np.mean(fcd_real_arr))
	# print('Var_real_fcd: ',np.var(fcd_real_arr))

	fcd_gen_arr = []
	for j in range(0,3):

	    print('Real_gen_iter: ',j)

	    real_data = real_data_fixed.copy()

	    np.random.shuffle(real_data)

	    real_data = real_data[:real_data.shape[0]//2]

	    real_data = real_data[:(real_data.shape[0]//batch_size)*batch_size]

	    noise_batch=mnist_nemgan.fixed_noise_gen(batch_size=real_data.shape[0])

	    alpha_batch = next(mnist_nemgan.fixed_dist_gen(batch_size=real_data.shape[0]))

	    gen_data = mnist_nemgan.generator.predict([alpha_batch,noise_batch],batch_size = batch_size)[0]

	    real_act = classifier.predict(real_data)

	    gen_act = classifier.predict(gen_data)
	    #gen_act = classifier.predict(real_data)

	    print('Calculating FCD......')
	    fcd = tf.contrib.gan.eval.diagonal_only_frechet_classifier_distance_from_activations(tf.convert_to_tensor(real_act),tf.convert_to_tensor(gen_act) )

	    sess = tf.Session()

	    fcd_val = sess.run(fcd)

	    print('FCD: ', fcd_val)

	    fcd_gen_arr.append(fcd_val)

	    sess.close()
	fcd_gen_arr = np.asarray(fcd_gen_arr)

	print('Mean_gen_fcd: ',np.mean(fcd_gen_arr))
	print('Var_gen_fcd: ',np.var(fcd_gen_arr))

	print('##'*30)
