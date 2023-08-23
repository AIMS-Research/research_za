# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:44:09 2022

@author: ljeantet
"""

'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

Script to process the audio files downloaded from Xenocanto.
It will generate a training dataset saved in audio format and the spectrograms of the validation dataset.
The training dataset is saved in audio format to be processed (data-augmentation and balance) in a second step with the script : Data_augmentation_and_balance.py  

The training and validation dataset have to be previously downloaded from Zenodo: https://doi.org/10.5281/zenodo.7828148 

'''

# Set up the  working directory where the data and scripts are contained (can be dowloaded from Github)
import os
os.chdir('../BirdClassifier_metadata')


from Preprocessing_Xenocanto import *
import params 


# I - Preprocessing of the training dataset

## Paths to the folders containing audio and annotation files for the tranain dataset
folder='~/Audio_files/Training'
folder_annotation='~/Annotation/Training'


## File containing metadata of selected species' recordings from Xenocanto
database_file='Xenocanto_metadata_qualityA_selection.csv'


## Folder where processed data will be saved
out_dir='out'




## Parameters of spectrograms obtained from the params.py file
n_fft = params.n_fft # Hann window length
hop_length = params.hop_length
n_mels=params.n_mels
nyquist_rate =params.nyquist_rate
lowpass_cutoff =params.lowpass_cutoff
win_length=params.win_length
f_min = params.f_min 
f_max=params.f_max


## Parameters for the segmentation of the acoustic files
segment_duration=params.segment_duration
hop_chunck=params.hop_chunck


## Type of spectogram to implement
type_spec=params.type_spec
downsample_rate = params.downsample_rate

## Indicate if we want to keep the obtained segment in audio format or image (spectrogram)
## Basically, we keep the training segments in audio format to process data augmentation in second step (Data-augmentation-and-balance.py)
type_saved_data='audio'
verbose=True #Boolean value indicating whether to print verbose output.


## Call the function
pre_pro = Preprocessing_Xenocanto(folder, folder_annotation, out_dir, database_file, lowpass_cutoff, 
             downsample_rate,  segment_duration, hop_chunck,  
             type_spec, type_saved_data, n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length)



## Preprocess the training dataset
X_calls, X_meta, Y_calls = pre_pro.create_dataset(verbose)

## Save the results
pre_pro.save_data_to_pickle(X_calls, X_meta, Y_calls, Saved_X='X_Xenocanto_audio_training-pow', Saved_meta='X_meta_Xenocanto_audio_training-pow',Saved_Y='Y_Xenocanto_audio_training-pow')



# II- Preprocessing of the validation dataset

## Paths to the folders containing audio and annotation files for the validation dataset
folder='~/Audio_files/Validation'
folder_annotation='~/Annotation/Validation'
out_dir='out'


## Indicate if we want to keep the obtained segment in audio format or image (spectrogram)
##here we want to save spectrograms for the validation dataset
type_saved_data='image'

## Call the function
pre_pro = Preprocessing_Xenocanto(folder,folder_annotation, out_dir, database_file, lowpass_cutoff, 
             downsample_rate,  segment_duration, hop_chunck,  
             type_spec, type_saved_data,                
             n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length)

## Preprocess the training dataset
X_calls_val, X_meta_val, Y_calls_val = pre_pro.create_dataset(verbose)

## Save the results
pre_pro.save_data_to_pickle(X_calls_val, X_meta_val, Y_calls_val, Saved_X='X_Xenocanto_melspect_val-pow', Saved_meta='X_meta_Xenocanto_melspect_val-pow',Saved_Y='Y_Xenocanto_melspect_val-pow')


