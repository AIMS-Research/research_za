# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:16:09 2022

@author: ljeantet
"""

import os
import pickle
import numpy as np
import pandas as pd
import json
import shutil


import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import time
import datetime
import random
from random import randint
import itertools
import matplotlib.pyplot as plt
import librosa

from CNNNetwork_Xenocanto import *


class Training:
    def __init__(self, folder, folder_out, X_file_name, X_meta_file_name, Y_file_name, 
                 vocab_size, max_length, type_data,
                 model_name, conv_layers, conv_filters, dropout_rate,conv_kernel,max_pooling_size,fc_units_1,fc_units_2,epochs,batch_size):
        
        
        self.folder = folder
        self.folder_out=folder_out
        self.X_file_name=X_file_name
        self.X_meta_file_name=X_meta_file_name
        self.Y_file_name=Y_file_name
        
        self.type_data=type_data   
        self.model_name=model_name
        self.conv_layers=conv_layers
        self.conv_filters=conv_filters
        self.dropout_rate=dropout_rate
        self.conv_kernel=conv_kernel
        self.max_pooling_size =max_pooling_size
        self.fc_units_1=fc_units_1
        self.fc_units_2=fc_units_2
        self.epochs=epochs
        self.batch_size=batch_size
        
        
        self.num_instances=0
        
        #embedding
        self.vocab_size=vocab_size
        self.max_length=max_length
        
    def load_data_from_pickle(self):
        '''
        Load all of the spectrograms from a pickle file
                
        '''
        infile = open(os.path.join(self.folder, self.X_file_name),'rb')
        X = pickle.load(infile)
        infile.close()
                
        infile = open(os.path.join(self.folder, self.X_meta_file_name),'rb')
        X_meta = pickle.load(infile)
        infile.close()
                
                
        infile = open(os.path.join(self.folder, self.Y_file_name),'rb')
        Y = pickle.load(infile)
        infile.close()

        return X, X_meta, Y
    
            
    def load_dico(self, name, key_int=False, print_me=False):
        
        with open(self.folder+name) as f:
            dico_str = json.loads(json.load(f))
            
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
                
        dico={conv_key(k):v for k,v in dico_str.items()}
            
        if print_me:
            print(dico)
                
        return dico
    
  
    
    def save_X_to_pickle(self, X, Saved_X='X_Picidae-pow'):
            '''
            Save all of the spectrograms to a pickle file.
        
            '''
            outfile = open(os.path.join(self.folder, Saved_X+'.pkl'),'wb')
            pickle.dump(X, outfile, protocol=4)
            outfile.close()
        


    
    def add_extra_dim(self, X):
        '''
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        '''
        X_new = np.reshape(X,(X.shape[0],
                                   X.shape[1],
                                   X.shape[2],1))
        return X_new
    
    def Embedding(self, X, X_meta, Y):
        
        'remove empty data => problem to fix'
        
        

        index=np.where([not np.any(x) for x in X_meta])

        Y=np.delete(Y,index, axis=0)
        X=np.delete(X,index, axis=0)
        X_meta=np.delete(X_meta,index, axis=0)


        ctn=[x[4] for x in X_meta]
        #replace each noun in continent by a number between 0 and 50 randomly selected
        encoded_docs = [keras.preprocessing.text.one_hot(d, self.vocab_size) for d in ctn]
        
        #generate our X_meta data with each noun of continent replaced by the number
        #mex_length max noun in the continent name , here =2 for United Kingdom and Russian Federation
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        X_meta=padded_docs
        

        return X, X_meta, Y
        
    
    def create_new_folder(self, k):
        
        now = datetime.datetime.now()
        today=now.strftime("%Y_%m_%d")
        dir_out=self.folder_out+"model="+today+"_"+self.model_name+"_"+self.type_data+"_"+str(k)
        
       
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
            print("we clear the directory:",dir_out)
        else:
            print("we create the directory:",dir_out)
    
        """création des dossiers """
        os.makedirs(dir_out)
        return dir_out
    
    def Preprocessing_balanced_dataset_base_line(self, X, X_meta, Y):
        
        Y_id=Y.copy()
        categories_ds = []
        categories_weights = []
        Y=to_categorical(Y)
        
        for species in np.unique(Y_id):
            #get back species data 
            index_sp=np.where(Y_id==species)[0]
            X_meta_species=X_meta[index_sp]
            X_species=X[index_sp]
            Y_species=Y[index_sp]
            
            #for each country
            countries=[x[4] for x in X_meta_species]
            for country in np.unique(countries):
                index=[x for x, z in enumerate(countries) if z == country]
                cat_metadata=X_meta_species[index]
                cat_x=X_species[index]
                cat_y=Y_species[index]
                num_instances_cat=X_meta_species[index].shape[0]
                
                cat_x=self.add_extra_dim(cat_x)
                
                
                cat_ds=tf.data.Dataset.from_tensor_slices((cat_x, cat_y))
                
                cat_ds = cat_ds.shuffle(num_instances_cat).repeat()
                cat_weight = float(min(num_instances_cat, self.max_instances_per_class))
                categories_weights.append(cat_weight)      
                categories_ds.append(cat_ds)


        dataset = tf.data.experimental.sample_from_datasets(
                                                        categories_ds,
                                                        weights=categories_weights)
        self.num_instances = int(sum(categories_weights))
        
        return dataset
    
    
    def Preprocessing_balanced_dataset_custom_CNN(self, X, X_meta, Y):
        
        
        Y_id=Y.copy()
        categories_ds = []
        categories_weights = []
        Y=to_categorical(Y)


        for species in np.unique(Y_id):
            #get back species data 
            index=np.where(Y_id==species)[0]
            X_meta_species=X_meta[index]
            X_species=X[index]
            Y_species=Y[index]
            
            #for each country
            countries=[x[4] for x in X_meta_species]
            for country in np.unique(countries):
                index=[x for x, z in enumerate(countries) if z == country]
                cat_meta=X_meta_species[index]
                cat_x=X_species[index]
                cat_y=Y_species[index]
                num_instances_cat=X_meta_species[index].shape[0]
                
                cat_x, cat_meta_emb, cat_y = self.Embedding(cat_x, cat_meta, cat_y)
                
                cat_x=self.add_extra_dim(cat_x)
                cat_meta_emb=np.reshape(cat_meta_emb,(cat_meta_emb.shape[0],
                                           cat_meta_emb.shape[1],1))
                
                cat_ds=tf.data.Dataset.from_tensor_slices(({"audio_input": cat_x, "meta_input": cat_meta_emb}, {"class_output": cat_y}))

                       
                cat_ds = cat_ds.shuffle(num_instances_cat).repeat()
                cat_weight = float(min(num_instances_cat, self.max_instances_per_class))
                categories_weights.append(cat_weight)      
                categories_ds.append(cat_ds)


        dataset = tf.data.experimental.sample_from_datasets(
                                                        categories_ds,
                                                        weights=categories_weights)


        
        self.num_instances = int(sum(categories_weights))
        
        return dataset
      
        
    def train(self):
        
        'load data'
        X_audio, X_meta, Y = self.load_data_from_pickle()
        
    
        'balance silence category'
        Y=to_categorical(Y)
        
        print(X_audio.shape)
        print(Y.shape)
        print(X_meta.shape)
            
        'Embedding'
        X_audio, X_meta, Y = self.Embedding(X_audio, X_meta, Y, 50, 2)
        X_audio=self.add_extra_dim(X_audio)
        'split the data'
        X_train, X_val, X_meta_train, X_meta_val, Y_train, Y_val= train_test_split(X_audio, X_meta, Y, 
                                                                shuffle = True, random_state = 42, train_size = 0.8)
        
        
        "load the model"
        seed=random.randint(1, 1000000)
        # Call backs to save weights
        dir_out=self.create_new_folder()  #attention :supprime donnees enregistrees si deja existant
        filepath= dir_out+"/weights_{}.hdf5".format(seed)
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
        
        networks = CNNNetwork(X_audio.shape[1], X_audio.shape[2], self.conv_layers, self.conv_filters,self.dropout_rate, self.conv_kernel, self.max_pooling_size,
                              self.fc_units_1, self.fc_units_2, self.epochs, self.batch_size)
        if self.model_name=='Base_line':
            model=networks.CNN_network()
        if self.model_name=='Custom_Meta_CNN':
            model=networks.custom_CNN_network()
            
    
        'training'
        start = time.time()
        
        if self.model_name=='Base_line':
            history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                              batch_size=32,
                              epochs=50,
                              verbose=2, 
                              callbacks=[checkpoint])
        
        if self.model_name=='Custom_Meta_CNN': 
            history = model.fit([X_train, X_meta_train], Y_train, validation_data=([X_val, X_meta_val], Y_val), 
                          batch_size=32,
                          epochs=50,
                          verbose=2, 
                          callbacks=[checkpoint])
        end = time.time()
        
        print('time training :', end-start)
        
        return history
     
    
    def get_dataset_partitions_tf(self, ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + val_split + test_split ) == 1
    
        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = ds.shuffle(shuffle_size, seed=12)
    
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
    
    
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds= ds.skip(train_size).skip(val_size)
    
    
        return train_ds, val_ds , test_ds
        
    
    def plot_loss(self, history, dir_out, saved=True):
        
        loss_values = history.history['loss']
        val_loss_values=history.history['val_loss']
        epochs = range(1, len(loss_values)+1)

        plt.plot(epochs, loss_values, label='Training Loss')
        plt.plot(epochs, val_loss_values, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        if saved==True:
            plt.savefig(dir_out+'/plot_Loss.png')
        
        
        acc_values=history.history['accuracy']
        val_acc_values=history.history['val_accuracy']
        plt.clf()
        plt.plot(epochs, acc_values, label='Training accuracy')
        plt.plot(epochs, val_acc_values, label='Val accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        if saved==True:
            plt.savefig(dir_out+'/plot_accuracy.png')
    
    
    def plot_confusion_matrix(self, dir_out, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="jet",
                          precision=2, saved=True
                         ):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
    
        
        fig, ax = plt.subplots(figsize=(12,12))
        ax.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.'+str(precision)+'f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "coral")
        fig.tight_layout()  
        
        if saved==True:
            fig.savefig(dir_out+"/"+title+".png")