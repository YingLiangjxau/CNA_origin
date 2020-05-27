# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:39:35 2019

@author: Administrator
"""
from keras.layers import Input,Dense
from keras.models import Model
#from keras import initializers


def autoencoder(X,n_features_bef,n_features_aft):
    
    input_img=Input(shape=(n_features_bef,))
    encoded=Dense(4096,kernel_initializer='random_uniform',activation='relu')(input_img)
    encoded=Dense(1024,activation='relu')(encoded)
    encoded=Dense(256,activation='relu')(encoded)
    encoded=Dense(n_features_aft,activation='relu')(encoded)

    decoded=Dense(256,activation='relu')(encoded)
    decoded=Dense(1024,activation='relu')(decoded)
    decoded=Dense(4096,activation='relu')(decoded)
    decoded=Dense(n_features_bef,activation='tanh')(decoded)


    autoencoder=Model(inputs=input_img,outputs=decoded)
    encoder=Model(inputs=input_img,outputs=encoded)

    autoencoder.compile(optimizer='adam',loss='mse')
    autoencoder.fit(X,X,epochs=16,batch_size=64,shuffle=True)
    X=encoder.predict(X)
    return X


def autoencoder_y(X,n_features_bef,n_features_aft,Y):
    input_img=Input(shape=(n_features_bef,))
    encoded=Dense(4096,kernel_initializer='random_uniform',activation='relu')(input_img)
    encoded=Dense(1024,activation='relu')(encoded)
    encoded=Dense(256,activation='relu')(encoded)
    encoded=Dense(n_features_aft,activation='relu')(encoded)

    decoded=Dense(256,activation='relu')(encoded)
    decoded=Dense(1024,activation='relu')(decoded)
    decoded=Dense(4096,activation='relu')(decoded)
    decoded=Dense(n_features_bef,activation='tanh')(decoded)


    autoencoder=Model(inputs=input_img,outputs=decoded)
    encoder=Model(inputs=input_img,outputs=encoded)

    autoencoder.compile(optimizer='adam',loss='mse')
    autoencoder.fit(X,X,epochs=16,batch_size=64,shuffle=True)
    X=encoder.predict(X)
    Y=encoder.predict(Y)
    return X,Y
