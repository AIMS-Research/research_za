# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:19:31 2022

@author: ljeantet


Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

code adapted from @author:  Copyright 2021 Fagner Cunha

from github https://github.com/alcunha/geo_prior_tf/blob/master/geo_prior/dataloader.py


"""

import os
os.chdir('../Geographical_prior')

from Prior_geo_Xenocanto_Preprocessing import *
from FCNet_model import *
import losses

import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from absl import app



folder='Data/'   #folder where we can find the dataset
database='Xenocanto_metadata_allrecordings_qualityA.csv'  #name of the training dataset
validation='Xenocanto_metadata_allrecordings_qualityB.csv' #name of the validation dataset

out_dir='Models_out/' #name of the folder where to save the weights of the model

num_classes=8994  #number of species present in the database : 8994 in total but only 8836 species with lat, lon information provided 

loc_encode='encode_cos_sin'
num_inputs=4

embed_dim=256
use_batch_normalization=False
lr=0.0005 #Initial learning rate
lr_decay=0.98
epochs=10
random_seed=42
batch_size=32
max_instances_per_class=50
dico=True

def build_input_data(folder, database, loc_encode, dico, batch_size, is_training, num_classes=None):
    
    pre_pro=Prior_geo_MetaData_Generator(folder, database,  
                                         loc_encode, 
                                         dico,
                                         batch_size, 
                                         max_instances_per_class=(max_instances_per_class if is_training \
                                                                                                else -1),
                                         is_training=is_training,
                                         num_classes=num_classes)


    return pre_pro.make_meta_dataset()
    
   

def lr_scheduler(epoch, lr):
  if epoch < 1:
      return lr
  else:
      return lr * lr_decay


def train_model(model,
                dataset,
                num_train_instances,
                val_dataset,
                loss_o_loc):
  
    
    
      
    #to save the weights of the model
    seed=random.randint(1, 1000000)
    dir_out=create_new_folder(out_dir, 'FCNet')  #warning : 
    filepath= dir_out+"/weights_{}.hdf5".format(seed)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,verbose=1, save_weights_only=True,save_freq=2*batch_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    

    model.compile(optimizer=optimizer,loss=loss_o_loc)

    return model.fit(dataset,
                   epochs=epochs,
                   steps_per_epoch=int(num_train_instances/batch_size),
                   callbacks=[cp_callback],
                   validation_data=val_dataset)


def set_random_seeds():
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def main(_):
    
    global num_classes
    set_random_seeds()
    
    dataset, num_instances, num_classes = build_input_data(folder, database, loc_encode, dico, batch_size, is_training=True, num_classes=num_classes)
    
    randgen = RandSpatioTemporalGenerator(
      loc_encode=loc_encode)

    val_dataset, _, _ = build_input_data(folder, validation, loc_encode, dico, batch_size, is_training=False, num_classes=num_classes)

    model = FCNet(num_inputs=4,
                embed_dim=embed_dim,
                num_classes=num_classes,
                rand_sample_generator=randgen,
                use_bn=use_batch_normalization)

    loss_o_loc = losses.weighted_binary_cross_entropy(pos_weight=num_classes)
    

    model.build((None, num_inputs))
    model.summary()

    train_model(model, dataset, num_instances, val_dataset, loss_o_loc)

if __name__ == '__main__':
    app.run(main)






