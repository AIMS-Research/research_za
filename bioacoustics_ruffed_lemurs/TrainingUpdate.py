# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:32:15 2022

@author: ljeantet
"""




import random
from DataBank import *
import librosa
import librosa.display
import time

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import gc
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam


starting_time = time.time()

seeds = [83654767,684646082,712788073,526487888,366517229,329376440,
         335643274,844525047,925265988,854844133,181041491,332143690,
         200463851,426285331,1109394976]

seeds = [712788074]

meta = 'Lemurs#0.30#100#56#3 input channels: **2, **3, **4. Normalise to [0,1]. No input_preprop()#'
INPUT_SHAPE = (128,151, 3)
EPOCHS = 5
save_results_folder = 'Lemurs/Results_Lemurs'


seed=seeds[0]
X_file_name="X_train_lj_4"
Y_file_name="Y_train_lj_4"

gc.collect()
db = DataBank('Lemurs', 0.30, 1, 'pow', X_file_name, Y_file_name, save_results_folder)

db.set_seed(seed)
    
X_train, X_val, Y_train, Y_val, new_seed = db.get_data(5500,  'roar')

print(X_train.shape, X_val.shape)
gc.collect()

dir_out=db.create_new_folder()

f=open(dir_out+'/'+str(seed)+"-A9-1.txt", "a+")
f.write(meta+'***\n')
f.write(str(seed)+'\n')
    
base_model = ResNet152V2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=INPUT_SHAPE,
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

# ------------ TRAIN ---------------------------
# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = Input(shape=INPUT_SHAPE)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(inputs, training=False)

x = Flatten()(x)
outputs = Dense(2, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

filepath=dir_out+'/'+"A9-1-{}.hdf5".format(seed)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
    
#history = model.fit(X_train, Y_train, validation_split=0.1,epochs=EPOCHS, callbacks=callbacks_list)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=EPOCHS, callbacks=callbacks_list)


 # ------------ FINE-TUNE ---------------------------
base_model.trainable = True
model.summary()

model.compile(
        optimizer=Adam(1e-5),  # Low learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

#history = model.fit(X_train, Y_train, validation_split=0.1,epochs=EPOCHS, callbacks=callbacks_list)    
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=EPOCHS, callbacks=callbacks_list)
    
model = tf.keras.models.load_model(filepath)

# Train
#cm = confusion_matrix(y_true=np.argmax(Y_train,axis=1), y_pred=np.argmax(model.predict(X_train), axis=1))
#cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print (cm)
del X_train, Y_train
print("done")

duration=time.time()-starting_time

print("Training took {:.2f} seconds".format(duration))
print("Which is {:.2f} minutes".format(duration/60))


Y_pred=np.argmax(model.predict(X_val), axis=1)
# Test
cm = confusion_matrix(y_true=np.argmax(Y_val,axis=1), y_pred=Y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print (cm)

f1 = f1_score(np.argmax(Y_val,axis=1), Y_pred)
print(f1)
#np.save(save_results_folder+"/Y_train",np.argmax(Y_train,axis=1))
#np.save(save_results_folder+"/Y_train_pred",np.argmax(model.predict(X_train), axis=1))

np.save(dir_out+"/Y_val",np.argmax(Y_val,axis=1))
np.save(dir_out+"/Y_val_pred",Y_pred)

duration=time.time()-starting_time

print("Training + Prediction took {:.2f} seconds".format(duration))
print("Which is {:.2f} minutes".format(duration/60))

