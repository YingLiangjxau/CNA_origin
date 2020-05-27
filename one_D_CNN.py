# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:36:14 2019

@author: Administrator
"""


from keras.layers import Conv1D,MaxPooling1D,Dropout,Reshape,concatenate,Flatten,AveragePooling1D
from keras.layers import Input,Dense
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau

def one_D_CNN(train_x,train_y,test_x,test_y,n_features,n_class):
    reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=2,mode='auto')
    input_img=Input(shape=(n_features,))
    layer_1=Reshape((n_features,1))(input_img)
    layer_1=Conv1D(128,7,padding='same',activation='relu')(layer_1)
    layer_1=MaxPooling1D(2,padding='same')(layer_1)
    layer_2=Conv1D(192,5,padding='same',activation='relu')(layer_1)
    layer_2=MaxPooling1D(2,padding='same')(layer_2)
    branch1=Conv1D(64,1,padding='same',activation='relu')(layer_2)
    branch2=Conv1D(48,1,padding='same',activation='relu')(layer_2)
    branch2=Conv1D(64,5,padding='same',activation='relu')(branch2)
    branch3=Conv1D(48,1,padding='same',activation='relu')(layer_2)
    branch3=Conv1D(64,7,padding='same',activation='relu')(branch3)
    branch4=Conv1D(48,1,padding='same',activation='relu')(layer_2)
    branch4=Conv1D(64,3,padding='same',activation='relu')(branch4)
    branchpool=MaxPooling1D(3,padding='same',strides=1)(layer_2)	
    branchpool=Conv1D(64,1,padding='same',activation='relu')(branchpool)
    layer_3=concatenate([branch1,branch2,branch3,branch4,branchpool])
    branch5=Conv1D(64,1,padding='same',activation='relu')(layer_3)
    branch6=Conv1D(48,1,padding='same',activation='relu')(layer_3)
    branch6=Conv1D(64,5,padding='same',activation='relu')(branch6)
    branch7=Conv1D(48,1,padding='same',activation='relu')(layer_3)
    branch7=Conv1D(64,7,padding='same',activation='relu')(branch7)
    branch8=Conv1D(48,1,padding='same',activation='relu')(layer_3)
    branch8=Conv1D(64,3,padding='same',activation='relu')(branch8)
    branchpool_2=MaxPooling1D(3,padding='same',strides=1)(layer_3)
    branchpool_2=Conv1D(64,1,padding='same',activation='relu')(branchpool_2)
    layer_4=concatenate([branch5,branch6,branch7,branch8,branchpool_2])
    layer_5=AveragePooling1D(3,padding='same')(layer_4)
    layer_5=Flatten()(layer_5)
    layer_5=Dropout(0.4)(layer_5)
    layer_6=Dense(128,activation='relu')(layer_5)
    layer_7=Dense(n_class,activation='softmax')(layer_6)
    model_classification=Model(inputs=input_img,outputs=layer_7)
    
    
    model_classification.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model_classification.fit(train_x,train_y,epochs=12,batch_size=16,validation_data=(test_x,test_y),callbacks=[reduce_lr])
    scores=model_classification.evaluate(test_x,test_y)
    y_poss=model_classification.predict(test_x)
    
    
    return scores,y_poss




