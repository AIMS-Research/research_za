# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:16:09 2022

@author: ljeantet
"""

'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This class is designed to facilitate training a model using a specified dataset. 
It provides methods for loading data, preprocessing, creating a new folder for storing output, splitting the dataset, plotting loss and accuracy, and plotting a confusion matrix.


'''


import os
import pickle
import numpy as np
import pandas as pd
import json
import shutil
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


import time
import datetime



class Training:
    def __init__(self, folder, folder_out, X_file_name, X_meta_file_name, Y_file_name, 
                 vocab_size, max_length, type_data, model_name):
        
        
        self.folder = folder
        self.folder_out=folder_out
        self.X_file_name=X_file_name
        self.X_meta_file_name=X_meta_file_name
        self.Y_file_name=Y_file_name
        
        self.type_data=type_data   
        self.model_name=model_name
        

        
        self.num_instances=0
        
        #embedding
        self.vocab_size=vocab_size
        self.max_length=max_length
        
    def load_data_from_pickle(self):
        '''
        
        Load all of the spectrograms from a pickle file.

        Returns:
            X (object): The loaded spectrograms.
            X_meta (object): The loaded metadata.
            Y (object): The loaded labels.

                
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
        
        """
        Load a dictionary from a JSON file.
    
        Parameters:
            -path (str): Path to the JSON file.
            -key_int (bool): If True, convert dictionary keys to integers. Otherwise, keep them as strings. Default is False.
            -print_me (bool): If True, print the loaded dictionary. Default is False.
    
        Returns:
            dict: Loaded dictionary.
        """
        
    
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
    
  
   
    def add_extra_dim(self, X):
        """
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.

        Args:
            X (ndarray): The input data to be reshaped.

        Returns:
            ndarray: The reshaped data with an additional dimension.
        """
        
        X_new = np.reshape(X,(X.shape[0],
                                   X.shape[1],
                                   X.shape[2],1))
        return X_new
    
    def Embedding(self, X, X_meta, Y):
        
        """
        Replacing continent nouns with encoded numbers. This encoded numbers will be used as input 2 by the Embedding layer in the multi-branch. 

        Args:
            X (ndarray): The input data.
            X_meta (ndarray): The metadata associated with the input data.
            Y (ndarray): The target labels.

        Returns:
            ndarray: The updated input data.
            ndarray: The embedded metadata.
            ndarray: The updated target labels.
        """
        
        

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
        
    
    def create_new_folder(self):
        
        """
        Creates a new folder for storing the output.
        Warning: this will clean up the directory if it already exists!

        Args:
            k (int): A parameter value used in the folder name.

        Returns:
            str: The path of the newly created folder.
        """
        
        # Retrieve today's date to create a folder with this date as identifier 
        now = datetime.datetime.now()
        today=now.strftime("%Y_%m_%d")
        dir_out=self.folder_out+"model="+today+"_"+self.model_name+"_"+self.type_data
        
        # Clear the directory if it already exists
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
            print("we clear the directory:",dir_out)
        else:
            print("we create the directory:",dir_out)
    
        # Creation of the folder
        os.makedirs(dir_out)
        return dir_out
    

    
    
    def get_dataset_partitions_tf(self, ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        
        """
        Splits a dataset into training, validation, and test partitions.

        Args:
            ds (tf.data.Dataset): The input dataset.
            ds_size (int): The size of the dataset.
            train_split (float, optional): The proportion of the dataset to allocate for training. Defaults to 0.8.
            val_split (float, optional): The proportion of the dataset to allocate for validation. Defaults to 0.1.
            test_split (float, optional): The proportion of the dataset to allocate for testing. Defaults to 0.1.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.
            shuffle_size (int, optional): The buffer size for shuffling. Defaults to 10000.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        """
        
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
        
        """
        Plots the training and validation loss as well as the training and validation accuracy.

        Args:
            history (keras.callbacks.History): The history object obtained from model training.
            dir_out (str): The directory path to save the plots.
            saved (bool, optional): Whether to save the plots. Defaults to True.
        """
            
        
        
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

        """
        Plots the confusion matrix.

        Args:
            dir_out (str): The directory path to save the plot.
            cm (numpy.ndarray): The confusion matrix.
            classes (list): The list of class labels.
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
            title (str, optional): The title of the plot. Defaults to 'Confusion matrix'.
            cmap (str, optional): The color map to use. Defaults to 'jet'.
            precision (int, optional): The precision of the values in the confusion matrix. Defaults to 2.
            saved (bool, optional): Whether to save the plot. Defaults to True.
        """
        
        
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
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

        
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