# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:31:31 2022

@author: ljeantet
"""

'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

Script to multiply the probabilities obtained from the Geographical Prior applied to metadata only (latitude and longitude) 
with the probabilities obtained from the basic model on the spectogram. 
Therefore it gives the final probability that a species corresponds to the call represented in the spectrogram knowing its location
'''

import os
from tensorflow import keras
import numpy as np
import tensorflow as tf
import math



## I- load model FCT 

#folder containing the different functions associated to the Geographical Prior model
os.chdir("../BirdClassifier_metadata/Geographical_prior")

from FCNet_model import *
from Prior_geo_Xenocanto_Preprocessing import *

# Path to the folder containing the weights of the Geographical prior model
ckpt_dir="../BirdClassifier_metadata/Geographical_prior/Models_out"

num_classes=8836

# Parameters of the Geographical prior model
loc_encode='encode_cos_sin'
embed_dim=256
use_batch_normalization=False
num_inputs=4


randgen = RandSpatioTemporalGenerator(
  loc_encode=loc_encode)


model_FCN= FCNet(num_inputs=4,
                embed_dim=embed_dim,
                num_classes=num_classes,
                rand_sample_generator=randgen,
                use_bn=use_batch_normalization)


model_FCN.build((None, num_inputs))
model_FCN.summary()

checkpoint_path = os.path.join(ckpt_dir, "weights_670488.hdf5") ## name of the weight file contained in the folder indicated above
model_FCN.load_weights(checkpoint_path)

### II - load Baseline CNN 

# Folder containing the different functions associated to Baseline model
os.chdir("../BirdClassifier_metadata")

from Training_class_Xenocanto import *
import params
from CNNetwork_Xenocanto import *

X_audio_shape_1=128
X_audio_shape_2=259

#parameters of the model
conv_layers = params.conv_layers
conv_filters = params.conv_filters
dropout_rate = params.dropout_rate
conv_kernel = params.conv_kernel
max_pooling_size = params.max_pooling_size 
fc_units_1 = params.fc_units_1



epochs = params.epoch
batch_size = params.batch_size

networks = CNNetwork(X_audio_shape_1, X_audio_shape_2, conv_layers, conv_filters, dropout_rate, conv_kernel, max_pooling_size,
                     fc_units_1, epochs, batch_size)

model_CNN=networks.CNN_network()

# Path to the folder containing the weights of the Baseline model selected
ckpt_dir="../BirdClassifier_metadata/Models_out/model=2022_12_14_Base_line_melspect"
checkpoint_path = os.path.join(ckpt_dir, "weights_684359.hdf5")

model_CNN.load_weights(checkpoint_path)

### III- Load of the validation dataset 

# Parameter of the embedding process contained in params file
vocab_size=params.vocab_size  #Size of the vocabulary
max_length=params.max_length # max number of words in the country
out_embedding=params.out_embedding #Dimension of the dense embedding


# Name of the model we want to appear in the name of the result folder 
model_name="Prior_geo"
 

type_data='melspect'  #between melspec, pcen,

# Path of the folder containing the validation dataset
validation_folder='../BirdClassifier_metadata/out/'

# Path of the folder to save the results
folder_out='../BirdClassifier_metadata/Models_out/'




# Name of the validation files
X_file_name='X_Xenocanto_melspect_val-pow.pkl'
X_meta_file_name='X_meta_Xenocanto_melspect_val-pow.pkl'
Y_file_name='Y_Xenocanto_melspect_val-pow.pkl'



trainer=Training(validation_folder,folder_out, X_file_name, X_meta_file_name, Y_file_name,
                  vocab_size, max_length, type_data, model_name)



X_val, X_meta_val, Y_val = trainer.load_data_from_pickle()


### IV- Predictions
#### 1- Prediction on acoutsic files with the Baseline model 

X_val=trainer.add_extra_dim(X_val)
Val_predict_CNN=model_CNN.predict(X_val)

#### 2- Prediction on the metadata with the Geographical prior

## preprocessing of the lat and long data
lat=np.asarray([x[0] for x in X_meta_val])
lng=np.asarray([x[1] for x in X_meta_val])

dataset = tf.data.Dataset.from_tensor_slices((
  lat,
  lng,
  Y_val))

loc_encode='encode_cos_sin'
def _encode_feat(feat, encode):
    if encode == 'encode_cos_sin':
        return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
    else:
        raise RuntimeError('%s not implemented' % encode)

    return feat 

def _preprocess_data(lat, lng, category_id):

    lat = tf.cond(lat!=0, lambda: lat/90.0, lambda: tf.cast(0.0, tf.float64))
    lng = tf.cond(lng!=0, lambda: lng/180.0, lambda: tf.cast(0.0, tf.float64))
    lat = _encode_feat(lat, loc_encode)
    lng = _encode_feat(lng, loc_encode)

    inputs = tf.concat([lng, lat], axis=0)
    inputs = tf.cast(inputs, tf.float32)

    outputs=category_id
    return inputs, outputs

dataset = dataset.map(_preprocess_data)

# Put the data into a numpy array
data=[]
Y=[]
for i, label in dataset:
    data.append(i.numpy())
    Y.append(label.numpy())
data=np.asarray(data)
Y=np.asarray(Y)

# Prediction with the Geographical Prior
Pred_FCT=model_FCN(data, training=False)

# Put the results into a numpy array
pred=[]
for i in Pred_FCT:
    pred.append(i.numpy())
preds=np.asarray(pred)

# We obtained predictions for 8836 species indexed on the basis of the 8994 species originally available on Xenocanto (158 species were associated with no lat-long information and have been removed).
preds.shape 

# Coordinate the dictonnaries of the two models 

labelInd_to_labelName_22=trainer.load_dico('labelInd_to_labelName_22species.json',key_int=False,print_me=False)
labelInd_to_labelName_8994=trainer.load_dico('labelInd_to_labelName_8994species.json',key_int=False,print_me=False)
labelName_to_labelInd_8994=trainer.load_dico('labelName_to_labelInd_8994species.json',key_int=False,print_me=False)
labelName_to_labelInd_22=trainer.load_dico('labelName_to_labelInd_22species.json',key_int=False,print_me=False)

# Find the index of the 22 species into the 8994 species available on Xenocanto
index=[]
for i in labelName_to_labelInd_22:
    index.append(labelName_to_labelInd_8994.get(i))


# Get back the outputs only for the 22 species selected and predicted by the Baseline model
pred_select=[]
for pred in preds:
    pred_select.append(pred[np.asarray(index)])
pred_select=np.asarray(pred_select)   




## Multiplie the prediction obtained with the Baseline model by the associated predictions obtained with the Geographical prior.
Predictions=pred_select*Val_predict_CNN



### V- Save the predictions and confusion matrix

dir_out=trainer.create_new_folder()  ##Warning: this will clean up the directory if it already exists!

np.save(dir_out+"/Y_val_pred",np.argmax(Predictions,1))
np.save(dir_out+"/Y_val",Y_val)

# Plot confusion matrix
cnf_matrix = confusion_matrix(Y_val, np.argmax(Predictions,1))
np.set_printoptions(precision=2)


class_names=[]
for i in np.unique(Y_val):
    class_names.append(labelInd_to_labelName_22[str(i)])

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
