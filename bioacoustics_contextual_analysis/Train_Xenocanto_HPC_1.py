# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:26:54 2022

@author: ljeantet
"""


from Trainer_Xenocanto import *
import params

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time



#training_folder='Xenocanto/out/'
#folder_out='Xenocanto/'

training_folder='out/'
folder_out='Models_out/'


memo=pd.Series()

X_file_name='X_Xenocanto_melspect_balanced_training_1-pow.pkl'
X_meta_file_name='X_meta_Xenocanto_melspect_balanced_training_1-pow.pkl'
Y_file_name='Y_Xenocanto_melspect_balanced_training_1-pow.pkl'


memo['X_train_name']=X_file_name
memo['X_meta_train_name']=X_meta_file_name
memo['Y_train_name']=Y_file_name

print(X_file_name)


#embedding
memo['vocab_size']=vocab_size=params.vocab_size
memo['max_length']=max_length=params.max_length

#parameters of the model
memo['conv_layers']=conv_layers = params.conv_layers
memo['fc_layers']=fc_layers = params.fc_layers
memo['max_pooling_size']=max_pooling_size = params.max_pooling_size 
memo['dropout_rate']=dropout_rate = params.dropout_rate
memo['conv_filters']=conv_filters = params.conv_filters
memo['conv_kernel']=conv_kernel = params.conv_kernel
memo['fc_units_1']=fc_units_1 = params.fc_units_1
memo['fc_units_2']=fc_units_2 = params.fc_units_2
memo['epochs']=epochs = params.epoch
memo['batch_size']=batch_size = params.batch_size



#model 

memo['model_name']=model_name="Custom_Meta_CNN_1" #between Base_line and Custom_Meta_CNN , Custom_Meta_CNN_1
print(model_name)
memo['type_data']=type_data='melspect'  #between melspec, pcen,


trainer=Training(training_folder,folder_out, X_file_name, X_meta_file_name, Y_file_name,
                  vocab_size,max_length,type_data,
               model_name, conv_layers, conv_filters, dropout_rate,conv_kernel,max_pooling_size,fc_units_1,fc_units_2,epochs,batch_size)




start_time = time.time()



'load data'
X_audio, X_meta, Y = trainer.load_data_from_pickle()
print('data loaded')

'load dictionary'
labelName_to_labelInd_country=trainer.load_dico('labelName_to_labelInd_country.json',key_int=False,print_me=False)
labelInd_to_labelName=trainer.load_dico('labelInd_to_labelName_22.json',key_int=False,print_me=False)


Y=to_categorical(Y)

if model_name=='Base_line':
    
    X_audio=trainer.add_extra_dim(X_audio)

    dataset=tf.data.Dataset.from_tensor_slices((X_audio, Y))
if model_name=='Custom_Meta_CNN':
    print('Embedding')
    X_audio, X_meta, Y = trainer.Embedding(X_audio, X_meta, Y)
    X_audio=trainer.add_extra_dim(X_audio)
    X_meta=np.reshape(X_meta,(X_meta.shape[0],
                           X_meta.shape[1],1))

    dataset=tf.data.Dataset.from_tensor_slices(({"audio_input": X_audio, "meta_input": X_meta}, {"class_output": Y}))
if model_name=='Custom_Meta_CNN_1':
    print('Associated number ')
    Country=[x[4] for x in X_meta]
    Country_id=[]
    for i in Country :
        Country_id.append(labelName_to_labelInd_country[i])

    Country_id=to_categorical(Country_id)

    X_audio=trainer.add_extra_dim(X_audio)

    dataset=tf.data.Dataset.from_tensor_slices(({"audio_input": X_audio, "meta_input": Country_id}, {"class_output": Y}))



#plt.imshow(train_iter_im.numpy()[:,:,0],  origin='lower') 

"""
if model_name=='Base_line':
    dataset=trainer.Preprocessing_balanced_dataset_base_line(X_audio, X_meta, Y)
if model_name=='Custom_Meta_CNN': 
    dataset=trainer.Preprocessing_balanced_dataset_custom_CNN(X_audio, X_meta, Y)
"""    
print(dataset)
"""
train_iter_im, train_iter_label = next(iter(dataset))
print (train_iter_im.numpy().shape, train_iter_label.numpy().shape)
print(train_iter_label.numpy())
print(train_iter_im.numpy())
"""
memo['X_audio_shape_0']=X_audio.shape[0]
memo['X_audio_shape_1']=X_audio_shape_1=X_audio.shape[1]
memo['X_audio_shape_2']=X_audio_shape_2=X_audio.shape[2]

del X_audio, X_meta, Y


print('split the data')
dataset_size=tf.data.experimental.cardinality(dataset).numpy()
#dataset_size=trainer.num_instances
print(dataset_size)
train_ds, val_ds, _ =trainer.get_dataset_partitions_tf(dataset, dataset_size, train_split=0.8, val_split=0.2, test_split=0.0)

del dataset


train_dataset = train_ds.repeat().batch(batch_size)
val_dataset = val_ds.repeat().batch(batch_size)


print("load the model")
"load the model"
memo['seed']=seed=random.randint(1, 1000000)
# Call backs to save weights
dir_out=trainer.create_new_folder(1)  #attention :supprime donnees enregistrees si deja existant
filepath= dir_out+"/weights_{}.hdf5".format(seed)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')



networks = CNNNetwork(X_audio_shape_1, X_audio_shape_2, conv_layers, conv_filters, dropout_rate, conv_kernel, max_pooling_size,
                     fc_units_1, fc_units_2, epochs, batch_size)
if model_name=='Base_line':
    model=networks.CNN_network()
    # model=networks.CNN_network_Lostanlen()
if model_name=='Custom_Meta_CNN':
    model=networks.custom_CNN_network()
    #model=networks.custom_CNN_network_Lostanlen()
if model_name=='Custom_Meta_CNN_1':
    model=networks.custom_CNN_network_1(Country_id.shape[1])
    #model=networks.custom_CNN_network_Lostanlen_1(Country_id.shape[1])
    
    
model.summary()    
    
print("training")


start = time.time()

history=model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          steps_per_epoch=(dataset_size*0.8)//batch_size, 
          validation_steps= (dataset_size*0.2)//batch_size,
          callbacks=[checkpoint])

"""
if model_name=='Base_line':
    print(model_name)
    history=model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          steps_per_epoch=(dataset_size*0.8)//BATCH_SIZE, 
          validation_steps= (dataset_size*0.2)//BATCH_SIZE,
          callbacks=[checkpoint])

if model_name=='Custom_Meta_CNN': 
    print(model_name)
    history=model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          steps_per_epoch=(dataset_size*0.8)//BATCH_SIZE, 
          validation_steps= (dataset_size*0.2)//BATCH_SIZE,
          callbacks=[checkpoint])
    
"""
    
end = time.time()


print('time training :', end-start)
print("--- %s seconds ---" % (time.time() - start_time))

print('Predict and Save on X_val')



#load validation data
memo['X_val_name']=X_file_name='X_Xenocanto_melspect_val_1-pow.pkl'
memo['X_meta_val_name']=X_meta_file_name='X_meta_Xenocanto_melspect_val_1-pow.pkl'
memo['Y_val_name']=Y_file_name='Y_Xenocanto_melspect_val_1-pow.pkl'


print(X_file_name)

trainer=Training(training_folder,folder_out, X_file_name, X_meta_file_name, Y_file_name,
                  vocab_size,max_length,type_data,
               model_name, conv_layers, conv_filters, dropout_rate,conv_kernel,max_pooling_size,fc_units_1,fc_units_2,epochs,batch_size)



X_val, X_meta_val, Y_val = trainer.load_data_from_pickle()


if model_name=='Base_line':
    X_val=trainer.add_extra_dim(X_val)
    Val_predict_CNN=model.predict(X_val)
if model_name=='Custom_Meta_CNN': 
    X_val, X_meta_val, Y_val = trainer.Embedding(X_val, X_meta_val, Y_val)
    X_val=trainer.add_extra_dim(X_val)
    X_meta_val=np.reshape(X_meta_val,(X_meta_val.shape[0],
                               X_meta_val.shape[1],1))
    Val_predict_CNN=model.predict([X_val, X_meta_val])
if model_name=='Custom_Meta_CNN_1': 
    
    X_val=trainer.add_extra_dim(X_val)
    
    Country=[x[4] for x in X_meta_val]
    Country_id_val=[]
    for i in Country :
        Country_id_val.append(labelName_to_labelInd_country[i])
    Country_id_val=to_categorical(Country_id_val)
    
    Val_predict_CNN=model.predict([X_val, Country_id_val])


#save
np.save(dir_out+"/Y_val_pred",Val_predict_CNN)
np.save(dir_out+"/Y_val",Y_val)


#validation
trainer.plot_loss(history, dir_out, saved=True)


cnf_matrix = confusion_matrix(Y_val, np.argmax(Val_predict_CNN,1))
np.set_printoptions(precision=2)



class_names=[]
for i in np.unique(Y_val):
    class_names.append(labelInd_to_labelName[str(i)])

trainer.plot_confusion_matrix(dir_out, cnf_matrix, classes=class_names, normalize=False,
                              title='Normalized confusion matrix')

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

Global_accuracy=(np.sum(TP)+np.sum(TN))/(np.sum(TP)+np.sum(TN)+np.sum(FP)+np.sum(FN))
print("Global Accuracy :", Global_accuracy)

#history=trainer.train()
print("                            ")
print("done")

memo.to_csv(dir_out+"/memo.csv")