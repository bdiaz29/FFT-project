import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.activations import *
import tensorflow.keras.backend as K


#custom metric to see how much percent error there is
def percent_error(y_true, y_pred):
    A=K.abs(K.mean(y_pred-y_true))
    B=K.abs(K.mean(y_pred))
    C=(A/B)
    return C
#returns random array
def rand_array():
    array=[]
    for i in range(256):
        array+=[random.randint(0,1024)]
    return array
#seperate real and imaginary components into one array 
def seperated(arr):
    array=[]
    for i in range(256):
        #real part
        array+=[np.real(arr[i])]
        #imagenary part
        array+=[np.imag(arr[i])]
    array=np.array(array)
    array=array
    return array

#prodces a random input and 
#the FFT of that input 
def random_in_out():
    a = rand_array()
    b = np.fft.fft(a)
    c = seperated(b)
    # which part
    return a ,c

#data generator to generate random inputs and outputs for FFTs
def generate_fft(batch_size=32):
    while True:
        batch_input = []
        batch_output = []
        for i in range(batch_size):
            input,output=random_in_out()
            batch_input+=[input]
            batch_output+=[output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield batch_x, batch_y




model=Sequential()
alpha=0.25
acti= tf.keras.layers.LeakyReLU(alpha=alpha)

model.add(Dense(512,input_dim=256))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
model.add(Dense(512))
model.add(Activation('sigmoid'))



model.compile(loss='mse',optimizer='adam',metrics=['mae',percent_error])

batchsize=16
samples=500000
steps_pre_epoch=int(samples/batchsize)

train=generate_fft(batch_size=32)
model.fit_generator(train,steps_per_epoch=steps_pre_epoch,epochs=5,verbose=2)



model.save("fft.h5")