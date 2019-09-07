import tensorflow as tf
import keras
import random
import scipy.io as sio
import numpy as np
import pickle
import tqdm
from keras.models import Sequential, Model
#from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
import random


full_classifier = keras.models.load_model('./cnns/cnn_mnist_10c.h5')
req_layer = 'flatten_1'
    #req_layer = 'final_logits'

classifier = Model(inputs=full_classifier.input,outputs=full_classifier.get_layer(req_layer).output)

print(full_classifier.summary())

with open("mnist_80_20_data.pkl",'rb') as pickle_file:
    data=pickle.load(pickle_file)/255.0
with open("sngan_images.pkl",'rb') as pickle_file:
    gen_data=pickle.load(pickle_file)

real_data=data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
print(real_data.shape)
gen_data=gen_data[:len(data)]
print(gen_data.shape)

plt.imshow(gen_data[0].reshape(28,28))
plt.show()

plt.imshow(real_data[0].reshape(28,28))
plt.show()

fcd_gen_arr = []
for j in range(0,50):

    print('Real_gen_iter: ',j)

    real_data_shuffled = real_data.copy()

    random.shuffle(real_data_shuffled)

    real_act = classifier.predict(real_data_shuffled)

    gen_act = classifier.predict(gen_data)

    print('Calculating FCD......')
    
    fcd = tf.contrib.gan.eval.diagonal_only_frechet_classifier_distance_from_activations(tf.convert_to_tensor(real_act),tf.convert_to_tensor(gen_act) )
    #fcd=tf.contrib.gan.eval.mean_only_frechet_classifier_distance_from_activations(tf.convert_to_tensor(real_act),tf.convert_to_tensor(gen_act))
    #fcd=tf.contrib.gan.eval.frechet_classifier_distance(
    #tf.convert_to_tensor(real_data.astype(np.float32)),tf.convert_to_tensor(gen_data.astype(np.float32)),
    #classifier)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    fcd_val = sess.run(fcd)

    print('FCD: ', fcd_val)

    fcd_gen_arr.append(fcd_val)

    sess.close()
fcd_gen_arr = np.asarray(fcd_gen_arr)

print('Mean_gen_fcd: ',np.mean(fcd_gen_arr))
print('Var_gen_fcd: ',np.var(fcd_gen_arr))

