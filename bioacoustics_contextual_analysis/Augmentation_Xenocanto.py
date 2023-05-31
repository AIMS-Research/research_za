# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:38:36 2022

@author: ljeantet
"""

"""
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This class contains the main functions to process data-augmentation on the training dataset in audio format.

"""



from Spectrogram_converter import *
import params

import glob, os
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from random import randint
import tensorflow_io as tfio
import json


## Need to load the spectrogram parameters from param file
n_fft = params.n_fft # Hann window length
hop_length = params.hop_length
n_mels=params.n_mels
nyquist_rate =params.nyquist_rate
lowpass_cutoff =params.lowpass_cutoff
win_length=params.win_length
f_min = params.f_min 
f_max=params.f_max


type_spec=params.type_spec #type of spectrogram we want to save the file : between 

## Call the fonction to convert audio files to spectrograms
conv_spectro=convertisseur_spectro(type_spec, n_fft, hop_length, n_mels, 
                               f_min, f_max, nyquist_rate, win_length )




class Balance_and_data_augmentation:
        def __init__(self, folder,X_file_name,  X_meta_file_name, Y_file_name, out_dir, sample_rate,
                          nb_sgts_ended=200, reduce=True, augmentation=True ):

            
            self.folder=folder
            self.X_file_name=X_file_name
            self.X_meta_file_name=X_meta_file_name
            self.Y_file_name=Y_file_name
            self.out_dir=out_dir
            self.sample_rate=sample_rate
        
            self.nb_sgts_ended=nb_sgts_ended
            self.reduce=reduce
            self.augmentation=augmentation
        
        
                   
    
        def load_data_from_pickle(self):
            '''
            Load the audio segments, metadata, and labels from a pickle file.

            Parameters:
                - path: Path to the directory containing the pickle files.
                - X: Name of the pickle file for audio segments.
                - X_meta: Name of the pickle file for metadata.
                - Y: Name of the pickle file for labels.

            Returns:
                - X: Loaded audio segments.
                - X_meta: Loaded metadata.
                - Y: Loaded labels.
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
     
        def time_shifting(self, X, X_meta, index):
            """
            Augment a segment of amplitude values by applying a time shift.

            Args:
                X (ndarray): Array of amplitude values.
                X_meta (ndarray): Array of corresponding metadata.
                index (list): List of indices of the files to choose from.

            Returns:
                tuple: Augmented segment and its metadata.
            """
            # Convert index to list
            index=list(index)
            # Randomly select an index from the given index list
            idx_pickup=random.sample(index, 1)
        
            # Retrieve the segment and metadata corresponding to the selected index
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]

        
            # Randomly select time into the segments
            random_time_point_segment = randint(1, len(segment)-1)

            # Apply time shift to the segment
            segment = self.time_shift(segment, random_time_point_segment)

          
            
            return segment, meta


        def time_shift(self, audio, shift):
            """
            Shift amplitude values to the right by a random value.

            The amplitude values are wrapped back to the left side of the waveform.

            Args:
                audio (ndarray): Array of amplitude values representing the audio waveform.
                shift (int): Amount of shift to apply to the waveform.

            Returns:
                ndarray: Augmented waveform with the shifted amplitude values.
            """
            
            augmented = np.zeros(len(audio))
            augmented [0:shift] = audio[-shift:]
            augmented [shift:] = audio[:-shift]
            
            return augmented


        def combining_same_class(self, X, X_meta, index):
            """
            Combine segments to create an augmented segment.

            Randomly selects two segments from the given indices and blends them to create a new segment.
            The blending weights are set to 0.6 and 0.4.

            Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

            Returns:
                tuple: Combined segment and its associated metadata.
            """
            # Convert index to list
            index=list(index)
        
            # Randomly select an index from the given index list
            idx_pickup=random.sample(index, 1)
            
            # Randomly select another file to combine with
            index.remove(idx_pickup)
            idx_combining=random.sample(index, 1)
        
            # combine the two files with different weights
            segment=self.blend(X[idx_pickup][0], X[idx_combining][0], 0.6, 0.4)
            
            #Retrieve the metadata corresponding to the selected index
            meta=X_meta[idx_pickup][0]   
     
            return segment , meta      
            
        def blend(self, audio_1, audio_2, w_1, w_2):
            """
            Blend two audio segments together using given weights.

            Takes two audio segments and blends them together using the provided weights.
            The blending weights determine the contribution of each segment in the resulting blended segment.

            Args:
                audio_1 (ndarray): First audio segment.
                audio_2 (ndarray): Second audio segment.
                w_1 (float): Weight for the first audio segment.
                w_2 (float): Weight for the second audio segment.

            Returns:
                ndarray: Blended audio segment.
            """
            
            
            augmented = w_1 * audio_1 + w_2 * audio_2
            return augmented 

        
        def add_noise_gaussian(self, X,X_meta,index):
            """
            Add Gaussian noise to an audio segment.

            Randomly selects an audio segment from the given indices and adds Gaussian noise to it.
            The noise is generated using a mean of 0 and a standard deviation of 0.009.

            Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

            Returns:
                tuple: Segment with added Gaussian noise and its associated metadata.
            """
            # Convert index to list
            index=list(index)
           
            # Randomly select an index from the given index list
            idx_pickup=random.sample(index, 1)
        
            # Retrieve the segment and metadata corresponding to the selected index
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]
        
            # Add Gaussian noise to the segment
            segment=segment+ 0.009*np.random.normal(0,1,len(segment))
            

            return segment, meta


        def implement_time_mask(self, X,X_meta, index ):
            """
            Implement a time mask on a spectrogram.

            Randomly selects an audio segment from the given indices, converts it into a spectrogram,
            and applies a time mask to the spectrogram.
            The time mask is generated using a parameter value of 100.

            Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

            Returns:
                tuple: Time-masked spectrogram and its associated metadata.
            """
            
            # Convert index to list
            index=list(index)
           
            # Randomly select an index from the given index list
            idx_pickup=random.sample(index, 1)
        
            # Retrieve the segment and metadata corresponding to the selected index
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]
            
             # Convert the segment into a spectrogram       
            spectro=conv_spectro.convert_single_to_image(segment, self.sample_rate)
              
            # Apply a time mask to the spectrogram
            time_mask = tfio.audio.time_mask(spectro, param=100)

            
            return time_mask.numpy(), meta
        

        def implement_freq_mask(self, X, X_meta, index ):
            """
            Implement a frequency mask on a spectrogram.

            Randomly selects an audio segment from the given indices, converts it into a spectrogram,
            and applies a frequency mask to the spectrogram.
            The frequency mask is generated using a parameter value of 100.

            Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

            Returns:
               tuple: Frequency-masked spectrogram and its associated metadata.
            """
            
            
            # Convert index to list
            index=list(index)
             
            # Randomly select an index from the given index list
            idx_pickup=random.sample(index, 1)
        
            # Retrieve the segment and metadata corresponding to the selected index
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]
            
            # Convert the segment into a spectrogram
            spectro=conv_spectro.convert_single_to_image(segment, self.sample_rate)
        
            # Apply a frequency mask to the spectrogram
            time_mask = tfio.audio.freq_mask(spectro, param=100)
           
            
            return time_mask.numpy(), meta

        
        def save_data_to_pickle(self, X, X_meta, Y, Saved_X='X_Picidae-pow', Saved_meta='X_meta_Picidae-pow',Saved_Y='Y_Picidae-pow'):
            '''
            Save the audio segments/spectrograms, metadata, and labels to pickle files.
        
            Parameters:
                    - X: Array of audio segments/spectrograms.
                    - X_meta: Array of corresponding metadata.
                    - Y: Array of labels.
                    - Saved_X: Name of the pickle file for audio segments/spectrograms.
                    - Saved_meta: Name of the pickle file for metadata.
                    - Saved_Y: Name of the pickle file for labels.
            '''
                
            outfile = open(os.path.join(self.out_dir, Saved_X+'.pkl'),'wb')
            pickle.dump(X, outfile, protocol=4)
            outfile.close()
        
            outfile = open(os.path.join(self.out_dir, Saved_meta+'.pkl'),'wb')
            pickle.dump(X_meta, outfile, protocol=4)
            outfile.close()
        
            outfile = open(os.path.join(self.out_dir, Saved_Y+'.pkl'),'wb')
            pickle.dump(Y, outfile, protocol=4)
            outfile.close()  
            
        
        def Generate_frequency_table(self, X, X_meta, Y, dico ):
            """
            Generate a frequency table by species and country.

            Args:
               X (ndarray): Input data containing segments.
               X_meta (ndarray): Metadata associated with the input data.
               Y (ndarray): Labels associated with the input data.
               dico (dict): Dictionary mapping label indices to species names.

            Returns:
               DataFrame: Frequency table showing species, countries, and frequencies.
            """
            
            
            # Create the DataFrame 
            meta=[x for x in X_meta]
            col_names=['lat','lon','date','time','cnt']
            Meta=pd.DataFrame(meta, columns=col_names)
            Meta['Label']=Y.copy()
            Meta['Exact_Label']=[dico[str(x)] for x in Meta.Label]

            # For each species, count the number of windows per country
            freq_df=[]
            for species in np.unique(Meta.Exact_Label):
        
                data=Meta[Meta.Exact_Label==species]
                ctn, count= np.unique(data.cnt, return_counts=True)
                for i,j in zip(ctn, count):
                    freq_df.append([species, i, j])
               

            col_name=['Species','Country','freq']   
            freq_db=pd.DataFrame(freq_df, columns=col_name )
        
            return freq_db
    
        def Balance_augment_and_reduce(self, X, X_meta, Y, dico, freq_db):
            """
            Balance, augment, and reduce the dataset based on frequency information.

            The function processes the frequency table and performs reduction or augmentation on the dataset based on the specified conditions.
            If the number of windows for a species and country combination is greater than the specified value `nb_sgts_ended` and reduction is enabled,
            random samples are removed from the dataset to match the desired number of segments.
            If the number of windows is smaller than `nb_sgts_ended` and augmentation is enabled,
            artificial samples are added using various methods such as time shifting, blending, adding noise, time masking, and frequency masking.

            Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                Y (ndarray): Labels associated with the input data.
                dico (dict): Dictionary mapping label indices to species names.
                freq_db (DataFrame): Frequency table showing species, countries, and frequencies.

            Returns:
                tuple: Augmented and reduced dataset with segments, metadata, and labels.
            """
            # Create storage lists
            X_spect=[]
            X_meta_spect=[]
            Y_spect=[]

            # Iterate over the frequency table 
            for i in range(freq_db.shape[0]):
                species=freq_db.Species[i]
                country=freq_db.Country[i]
                freq=freq_db.freq[i]
            
                Y_labels=[dico[x.astype(str)] for x in Y]
                Meta_cnt=[x[4] for x in X_meta]

                # If the number of windows for a species and country combination is greater than the specified value `nb_sgts_ended` and reduction is enabled
                if ((freq > self.nb_sgts_ended) & (self.reduce==True)) :
                    print(species, country)
                    print('nb of window > nb_sgts_ended ')
                    print(" Reduction")
                    index=np.where((np.asarray(Y_labels)==species) & (np.asarray(Meta_cnt)==country))[0]
                    
                    # Randomly select the number of windows to keep
                    index_to_keep=np.array(random.sample(list(index), self.nb_sgts_ended))
                
                    # Add the selected windows to the storage lists
                    X_spect.extend(conv_spectro.convert_all_to_image(X[index_to_keep], self.sample_rate))
                    X_meta_spect.extend(X_meta[index_to_keep])
                    Y_spect.extend(Y[index_to_keep])
                
                    # Remove the selected windows from the original data
                    X=np.delete(X, index, axis=0)
                    X_meta=np.delete(X_meta, index, axis=0 )
                    Y=np.delete(Y, index, axis=0 )
                
                # If the number of windows is smaller than `nb_sgts_ended` and augmentation is enabled
                if ((freq < self.nb_sgts_ended) & (self.augmentation==True)):
                    
                    print(species, country)
                    print('nb of window < nb_sgts_ended ')
                    print(" Augmentation")
                    
                    
                    index=np.where((np.asarray(Y_labels)==species) & (np.asarray(Meta_cnt)==country))[0]
                    
                    # Determine how many segments need to be added in total
                    nb_to_add=self.nb_sgts_ended-freq
                    # Determine how many segments need to be added per method
                    nb_to_augm_per_method=(nb_to_add//5)+1
                
                    # Convert and add the existing windows
                    X_spect.extend(conv_spectro.convert_all_to_image(X[index], self.sample_rate))
                    X_meta_spect.extend(X_meta[index])
                    Y_spect.extend(Y[index])
                
                    # Loop to process each augmentation method
                    for j in range(0,nb_to_augm_per_method):
                    
                        segment, meta= self.time_shifting(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, self.sample_rate))
                        X_meta_spect.append(meta)
                    
                        segment, meta= self.combining_same_class(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, self.sample_rate))
                        X_meta_spect.append(meta)
                    
                        segment, meta= self.add_noise_gaussian(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, self.sample_rate))
                        X_meta_spect.append(meta)
                    
                        spectro, meta=self.implement_time_mask(X, X_meta, index)
                        X_spect.append(spectro)
                        X_meta_spect.append(meta)
                    
                        spectro, meta=self.implement_freq_mask(X, X_meta, index)
                        X_spect.append(spectro)
                        X_meta_spect.append(meta)
                    
                    # Add corresponding labels for the augmented segments
                    Y_spect.extend((5*nb_to_augm_per_method)*[Y[index[0]]])
                
                    # Remove the original windows from the data
                    X=np.delete(X, index, axis=0)
                    X_meta=np.delete(X_meta, index, axis=0 )
                    Y=np.delete(Y, index, axis=0 )
                    
                    
            return X_spect, X_meta_spect, Y_spect    

            



