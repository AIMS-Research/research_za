# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:46:15 2022

@author: ljeantet
"""


'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

Script to balance the training dataset.

we either applied data augmentation or data reduction so that 200 segments were available per species and country. 
The reduction involved random sampling so that only 200 segments remain. 
For data augmentation, we artificially added new samples to reach 200 segments using five different methods, namely, time shifting, blending, adding noise, time and frequency masking.

The number of segments to achieve at the end of the reduction/augmentation process is manually chosen as parameter (nb_sgts_ended)


Each window of the training dataset wil be converted in spectrogram and the training dataset saved in pickle file 

'''

# Set up the  working directory where the data and scripts are contained
import os
os.chdir('../BirdClassifier_metadata')


from Augmentation_Xenocanto import *



## Path where the training dataset has been saved in pickle
folder='out/'
## Path for the folder where to save the balanced training dataset
out_dir='out/'



## Names of the files
X_file_name='X_Xenocanto_audio_training-pow.pkl
X_meta_file_name='X_meta_Xenocanto_audio_training-pow.pkl'
Y_file_name='Y_Xenocanto_audio_training-pow.pkl'


sample_rate = 22050  #sample rate after pre-processing. Audio files have already been downsampled in the previous step

#nb of windows that we want in the balance dataset per species and per country.
nb_sgts_ended=150

# Operations that we want to realise
reduce=True
augmentation=True

## Call the function
data_augmenter=Balance_and_data_augmentation(folder, X_file_name,  X_meta_file_name, Y_file_name, out_dir, sample_rate,
               nb_sgts_ended, reduce, augmentation )


## Load the data
print('load data and metadata')
X_audio, X_meta, Y_calls = data_augmenter.load_data_from_pickle()
#load dictionary containing the maps between the number and the species, normally created in previous step (Datapreparation_Xenocanto.py) and saved in the save folder than training datasat
labelInd_to_labelName=data_augmenter.load_dico('labelInd_to_labelName_22species.json',key_int=False,print_me=False)

## Visualize the number of windows per species and country before data augmentation
freq_db=data_augmenter.Generate_frequency_table( X_audio, X_meta, Y_calls, labelInd_to_labelName )
np.mean(freq_db.freq)

## Process data-augmentation and reduce the number of windows for over-represented categories to balance the dataset
print("generate datauagmentation and balancing and convert audio files into spectrograms")
X_spect, X_meta_spect, Y_spect= data_augmenter.Balance_augment_and_reduce(X_audio, X_meta, Y_calls, labelInd_to_labelName, freq_db)
 
   
print('put into arrays')
X_spect=np.asarray(X_spect)
X_meta_spect=np.asarray(X_meta_spect)
Y_spect=np.asarray(Y_spect)


## Visualize the number of windows per species and country after data augmentation
freq_db_1=data_augmenter.Generate_frequency_table( X_spect, X_meta_spect, Y_spect, labelInd_to_labelName )

## Save the data
print('save data')
data_augmenter.save_data_to_pickle(X_spect, X_meta_spect, Y_spect, Saved_X='X_Xenocanto_melspect_balanced_training-pow', Saved_meta='X_meta_Xenocanto_melspect_balanced_training-pow',Saved_Y='Y_Xenocanto_melspect_balanced_training-pow')


