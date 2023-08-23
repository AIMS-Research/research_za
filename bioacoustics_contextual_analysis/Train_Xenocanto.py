# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:26:54 2022

@author: ljeantet
"""

'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This is the main script to train and evaluate the differents models presented in the article  ; 

-Case I : baseline model - classic CNN
    model_name="Baseline" 
    
-Case II : One-hot encoding - two-branch CNN with spectrogram as input 1 and metadata as input 2 (country). 
   We assigned  a unique number (n = 28) to each country used in this study and converted the number into a one-hot encoded vector.
   model_name="Custom_Meta_CNN_1" 
   
-Case III : Metadata embedding - two-branch CNN with spectrogram as input 1 and metadata as input 2 (country)
   To integrate the country names in a vector of chosen dimension, we started by assigning a unique numerical value between 0 and 50 (vocab_size) to each word present in our country list. 
   Since some countries are composed of two words (e.g. South Africa, United Kingdom...), this results in a vector of size 2 (max_length), with a 0 in the second position for countries with only one word (e.g. Belgium, Venezuela, ..). 
   We subsequently incorporated an embedding layer that mapped each value into an 8-dimensional transformed space (out_embedding), resulting in a vector of size [2, 8] for each country.
   model_name="Custom_Meta_CNN"


The Case IV (geographical prior) requires the baseline model to be trained using this script (Case I), and the geographical prior to be trained using the script : XX.py.
The script XX.py is used to combine the two models and get the final results.

'''

# Set up the  working directory where the data and scripts are contained 
import os
os.chdir('../BirdClassifier_metadata')


from Training_class_Xenocanto import *
from CNNetwork_Xenocanto import *
import params

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time
import random




# Path of the folder containing the training and validation dataset
training_folder='out/'

# Path of the folder to save the results
folder_out='Models_out/'



# Memo to save all the parameters used 
memo=pd.Series()

# Names of the files
X_file_name='X_Xenocanto_melspect_balanced_training-pow.pkl'
X_meta_file_name='X_meta_Xenocanto_melspect_balanced_training-pow.pkl'
Y_file_name='Y_Xenocanto_melspect_balanced_training-pow.pkl'


memo['X_train_name']=X_file_name
memo['X_meta_train_name']=X_meta_file_name
memo['Y_train_name']=Y_file_name

print(X_file_name)


# Parameter of the embedding process contained in params file
memo['vocab_size']=vocab_size=params.vocab_size  #Size of the vocabulary
memo['max_length']=max_length=params.max_length # max number of words in the country
memo['out_embedding']=out_embedding=params.out_embedding #Dimension of the dense embedding

# Parameters of the baseline model contained in params file
memo['conv_layers']=conv_layers = params.conv_layers #number of convolution layers
memo['conv_filters']=conv_filters = params.conv_filters #number of filters (same parameter for every convolution layer if conv_layers>)
memo['conv_kernel']=conv_kernel = params.conv_kernel #size of the kernel (same parameter for every convolution layer if conv_layers>0)

memo['fc_units_1']=fc_units_1 = params.fc_units_1 # nb of units of the first fully-connected layer

memo['max_pooling_size']=max_pooling_size = params.max_pooling_size #integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
memo['dropout_rate']=dropout_rate = params.dropout_rate #Float between 0 and 1. Fraction of the input units to drop


memo['epochs']=epochs = params.epoch  
memo['batch_size']=batch_size = params.batch_size



# Choose the model that you want to apply between :
    ## 'Base_line' : classical CNN, Case I in the article 
    ## 'Custom_Meta_CNN_1' : Two bramch CNN taking spectrogram as input 1 and metadata as input 2 : country name is mapping into one-hot encoded vector (Case II)
    ## 'Custom_Meta_CNN_2' : Two-branch CNN taking spectrogram as input 1 and metadata as input 2 : country name is mapping into an embedding space of size 8 (Case III)
 

memo['model_name']=model_name="Custom_Meta_CNN_2" 
print('Model to be loaded : ',model_name)

# Type of visual reprensation of the acoustic files we are working with:
    ## 'spect' : The spectrogram representation of the audio
    ## 'melspect' : The mel-spectrogram representation of the audio.
    ## 'pcen' : The PCEN representation of the audio
memo['type_data']=type_data='melspect'  



## Call the class containing all the needed functions to prepare the data

trainer=Training(training_folder,folder_out, X_file_name, X_meta_file_name, Y_file_name,
                  vocab_size, max_length, type_data, model_name)




## I -  Load data

## To measure how long it takes to load the data and train the model
start_time = time.time()

X_audio, X_meta, Y = trainer.load_data_from_pickle()
print('data loaded')

'load dictionary'
# Dictionnary that maps one unique number to each country
labelName_to_labelInd_country=trainer.load_dico('labelName_to_labelInd_country.json',key_int=False,print_me=False)
# Dictionary that associates the identification number with each species
labelInd_to_labelName=trainer.load_dico('labelInd_to_labelName_22species.json',key_int=False,print_me=False)

Y=to_categorical(Y)

# PLace the data into tensor Dataset
if model_name=='Base_line':
    print('Case I : no metadata process')
    X_audio=trainer.add_extra_dim(X_audio)
    dataset=tf.data.Dataset.from_tensor_slices((X_audio, Y))

if model_name=='Custom_Meta_CNN_1':
    print('Case II : Process the metadata into a one-hot encoded vector ')
    
    #Get back the country name from the metadata 
    Country=[x[4] for x in X_meta]
    Country_id=[]
    # Associate each country with a unique number based on the previously established dictionary
    for i in Country :
        Country_id.append(labelName_to_labelInd_country[i])

    Country_id=to_categorical(Country_id)
    X_audio=trainer.add_extra_dim(X_audio)
    dataset=tf.data.Dataset.from_tensor_slices(({"audio_input": X_audio, "meta_input": Country_id}, {"class_output": Y}))
    
    
if model_name=='Custom_Meta_CNN_2':
    print('Case III : Process the metadata into a 2 size vector for the embedding layer')
    
    
    #Replacing every continent nouns with encoded numbers between 0 and 50. Each value in the obtained X_meta is now size 2.
    X_audio, X_meta, Y = trainer.Embedding(X_audio, X_meta, Y)
    
    X_audio=trainer.add_extra_dim(X_audio)
    X_meta=np.reshape(X_meta,(X_meta.shape[0],
                           X_meta.shape[1],1))

    dataset=tf.data.Dataset.from_tensor_slices(({"audio_input": X_audio, "meta_input": X_meta}, {"class_output": Y}))



print(dataset)

memo['X_audio_shape_0']=X_audio.shape[0]
memo['X_audio_shape_1']=X_audio_shape_1=X_audio.shape[1]
memo['X_audio_shape_2']=X_audio_shape_2=X_audio.shape[2]

del X_audio, X_meta, Y


print('Split the training dataset into a Training/Validation dataset with a ratio 0.8/0.2')
dataset_size=tf.data.experimental.cardinality(dataset).numpy()
train_ds, val_ds, _ =trainer.get_dataset_partitions_tf(dataset, dataset_size, train_split=0.8, val_split=0.2, test_split=0.0)

del dataset


train_dataset = train_ds.repeat().batch(batch_size)
val_dataset = val_ds.repeat().batch(batch_size)

## II - Load the Model




print("load the model")

memo['seed']=seed=random.randint(1, 1000000)

# Call backs to save weights
# Create within the save folder a specific folder associated to this expriment
#The name of the folder will contain the date, the model type and data type
dir_out=trainer.create_new_folder()  #Warning: this will clean up the directory if it already exists!

filepath= dir_out+"/weights_{}.hdf5".format(seed)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')


# Call the class contaning the different architectures
networks = CNNetwork(X_audio_shape_1, X_audio_shape_2, conv_layers, conv_filters, dropout_rate, conv_kernel, max_pooling_size,
                     fc_units_1, epochs, batch_size)

if model_name=='Base_line':
    model=networks.CNN_network()

if model_name=='Custom_Meta_CNN_1':
    model=networks.custom_CNN_network_1(Country_id.shape[1])


if model_name=='Custom_Meta_CNN_2':
    model=networks.custom_CNN_network_2(vocab_size, max_length, out_embedding)

    
model.summary()    
    


## III - Training

print("training")




history=model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          steps_per_epoch=(dataset_size*0.8)//batch_size, 
          validation_steps= (dataset_size*0.2)//batch_size,
          callbacks=[checkpoint])
    
end = time.time()


print('time training :', end-start_time)



## IV- Validation Dataset

# A- load validation data
memo['X_val_name']=X_file_name='X_Xenocanto_melspect_val-pow.pkl'
memo['X_meta_val_name']=X_meta_file_name='X_meta_Xenocanto_melspect_val-pow.pkl'
memo['Y_val_name']=Y_file_name='Y_Xenocanto_melspect_val-pow.pkl'


print(X_file_name)


trainer=Training(training_folder,folder_out, X_file_name, X_meta_file_name, Y_file_name,
                  vocab_size, max_length, type_data, model_name)


X_val, X_meta_val, Y_val = trainer.load_data_from_pickle()

# B - Preparation of the Validation Dataset and Prediction

if model_name=='Base_line':
    X_val=trainer.add_extra_dim(X_val)
    Val_predict_CNN=model.predict(X_val)



if model_name=='Custom_Meta_CNN_1': 
    
    X_val=trainer.add_extra_dim(X_val)
    Country=[x[4] for x in X_meta_val]
    Country_id_val=[]
    for i in Country :
        Country_id_val.append(labelName_to_labelInd_country[i])
    Country_id_val=to_categorical(Country_id_val) 
    Val_predict_CNN=model.predict([X_val, Country_id_val])

if model_name=='Custom_Meta_CNN_2': 
    X_val, X_meta_val, Y_val = trainer.Embedding(X_val, X_meta_val, Y_val)
    X_val=trainer.add_extra_dim(X_val)
    X_meta_val=np.reshape(X_meta_val,(X_meta_val.shape[0],
                               X_meta_val.shape[1],1))
    Val_predict_CNN=model.predict([X_val, X_meta_val])



# Save
np.save(dir_out+"/Y_val_pred",Val_predict_CNN)
np.save(dir_out+"/Y_val",Y_val)


## V- EValuate the model

# Plot of the validation and training loss
trainer.plot_loss(history, dir_out, saved=True)

# Calculation of the confusion matrix
cnf_matrix = confusion_matrix(Y_val, np.argmax(Val_predict_CNN,1))


# Plot of the confuion matrix
np.set_printoptions(precision=2)
class_names=[]
for i in np.unique(Y_val):
    class_names.append(labelInd_to_labelName[str(i)])

trainer.plot_confusion_matrix(dir_out, cnf_matrix, classes=class_names, normalize=False,
                              title='Normalized confusion matrix')


# Calculation of the Global accuracy and precision measurements
TP=np.diag(cnf_matrix)
FP=np.sum(cnf_matrix,axis=1)-TP
FN=np.sum(cnf_matrix,axis=0)-TP
TN=np.ones(len(class_names))*np.sum(cnf_matrix)-TP-FP-FN

Accuracy=(TP+TN)/(TP+TN+FP+FN)
Recall=TP/(TP+FN)
Precision =TP/(TP+FP)
Specificity =TN/(TN+FP)

df = pd.DataFrame({'Accuracy': Accuracy, 'Recall': Recall, 'Precision': Precision, 'Specificity': Specificity},
                      index = class_names)
df.to_csv(dir_out+'/Accuracy_measurements.csv')

Global_accuracy=(np.sum(TP)+np.sum(TN))/(np.sum(TP)+np.sum(TN)+np.sum(FP)+np.sum(FN))
print("Global Accuracy :", Global_accuracy)


print("                            ")
print("done")

memo.to_csv(dir_out+"/memo.csv")