# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:34:41 2022

@author: ljeantet
"""



'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This class is designed to preprocess audio data from the Xenocanto database. 
It includes various methods for filtering, downsampling, converting audio to different spectrogram representations (spectrograms, mel-spectrograms, and PCEN), 
and extracting metadata from annotation files.


'''



import glob, os
import numpy as np
import random
import librosa.display
import librosa
from xml.dom import minidom
from scipy import signal
from random import randint
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import math
import datetime
from os import listdir
from os.path import isfile, join
import os
import json



class Preprocessing_Xenocanto:
    
    def __init__(self, folder, folder_annotation, out_dir, database_file, lowpass_cutoff, 
                 downsample_rate,  segment_duration, hop_chunck,  
                 type_spec, type_saved_data, n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length ):
        
        self.folder = folder
        self.folder_annotation=folder_annotation
        self.database_file=database_file
        self.out_dir=out_dir
        self.segment_duration = segment_duration
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.hop_chunck=hop_chunck
        self.type_spec=type_spec
        self.type_saved_data=type_saved_data
        self.nyquist_rate = nyquist_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.win_length=win_length

        



    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        
        '''
        Butterworth lowpass filter design.
    
        Parameters:
            - cutoff: The cutoff frequency of the filter.
            - nyq_freq: The Nyquist frequency of the signal.
            - order: The order of the filter (default is 4).
    
        Returns:
            - b: Numerator coefficients of the filter transfer function.
            - a: Denominator coefficients of the filter transfer function.
        '''  
        
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        
        '''
        Apply Butterworth lowpass filter to the input data.
  
        Parameters:
            - data: The input signal or data to be filtered.
            - cutoff_freq: The cutoff frequency of the filter.
            - nyq_freq: The Nyquist frequency of the signal.
            - order: The order of the filter (default is 4).
  
        Returns:
            - y: The filtered signal or data after applying the Butterworth lowpass filter.
        '''
        
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def read_audio_file(self, file_path):
        '''
        Read an audio file and return the amplitudes and sample rate.

        Parameters:
                - file_path: The path to the audio file.

        Returns:
                - audio_amps: The amplitudes of the audio file.
                - audio_sample_rate: The sample rate of the audio file.
        '''
   
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(file_path, sr=None)
    
        return audio_amps, audio_sample_rate


    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        '''
        Downsample an audio file to a given new sample rate.

        Parameters:
            - amplitudes: The amplitudes of the audio file.
            - original_sr: The original sample rate of the audio file.
            - new_sample_rate: The desired new sample rate for downsampling.

        Returns:
            - downsampled_amplitudes: The downsampled amplitudes of the audio file.
            - new_sample_rate: The sample rate after downsampling.
        '''
        return librosa.resample(amplitudes, 
                                orig_sr=original_sr, 
                                target_sr=new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate


    def build_mel_spectro(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
    
        Parameters:
            - audio: The audio amplitudes.
    
        Returns:
            - S1: The mel-spectrogram representation of the audio.
        '''
        
        S = librosa.feature.melspectrogram(y=audio, n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
    
        return S1

    def build_spectro(self, audio):
        '''
        Convert amplitude values into a spectrogram.
    
        Parameters:
            - audio: The audio amplitudes.
    
        Returns:
            - S1: The spectrogram representation of the audio.
        '''
        
        D=librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        image=librosa.amplitude_to_db(abs(D))         
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
        return S1
    
    def build_pcen(self, audio, sample_rate):
        '''
        Convert amplitude values into a Per-Channel Energy Normalization (PCEN).
    
        Parameters:
            - audio: The audio amplitudes.
            - sample_rate: The sample rate of the audio.
    
        Returns:
            - pcen: The PCEN representation of the audio.
        '''     
        
        audio = (audio * (2 ** 31)).astype("float32")
        stft = librosa.stft(audio, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        abs2_stft = (stft.real * stft.real) + (stft.imag * stft.imag)
        melspec = librosa.feature.melspectrogram(y=None,S=abs2_stft,sr=sample_rate,n_fft=self.n_fft,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)

        pcen = librosa.pcen(melspec,sr=sample_rate, hop_length=self.hop_length, gain=0.8, bias=10, power=(1/4), time_constant=0.4, 
                     eps=1e-06)

        pcen = pcen.astype("float32")
        
        
        return pcen
    

    def convert_single_to_image(self,segment, sample_rate):    
        '''
        Convert a single audio segment into the chosen visual representation.
        
        Parameters:
            - segment: The audio segment.
            - sample_rate: The sample rate of the audio.
    
        Returns:
            - image: The visual representation of the audio segment.
        '''
    
        if self.type_spec=='spectro':
            image=self.build_spectro(segment)
        elif self.type_spec=='mel-spectro':
            image=self.build_mel_spectro(segment)
        elif self.type_spec=='pcen':
            image=self.build_pcen(segment, sample_rate)
        else :
            print('error')
    
        return image 


    
    def convert_all_to_image(self ,audio_list, sample_rate):
        '''
        Convert a number of segments into their corresponding visual representations.
    
        Parameters:
            - audio_list: A list of audio segments.
            - sample_rate: The sample rate of the audio.
    
        Returns:
            - spectrograms: An array of the visual representations (spectrograms) of the audio segments.
        '''
        spectrograms = []
        for segment in audio_list:
            spectrograms.append(self.convert_single_to_image(segment, sample_rate))

        return np.array(spectrograms)    


    def print_spectro(self, spectro, sample_rate, title="spectogram"):
        '''
        Plot the spectrogram.
    
        Parameters:
            - spectro: The spectrogram to be plotted.
            - sample_rate: The sample rate of the audio.
            - title: The title of the plot (default is "spectrogram").
        '''
        
        fig, ax = plt.subplots(figsize=(12,5))
        img=librosa.display.specshow(spectro,sr=sample_rate,hop_length=self.hop_length, cmap='magma', x_axis='time',ax=ax)
        fig.colorbar(img, ax=ax,format='%+2.0f dB')
        fig.suptitle(title)

    
    def create_and_save_dictionnary(self,labels, name_dict):
        '''
        Create and save dictionaries for label name to label index and label index to label name mappings.
    
        Parameters:
            - labels: A list of label names.
            - name_dict : indication to identify the dictionary
    
        Returns:
            - labelName_to_labelInd: A dictionary mapping label names to label indices.
            - labelInd_to_labelName: A dictionary mapping label indices to label names.
        '''
        
        
    
        labelName_to_labelInd={}
        labelInd_to_labelName={}
    
        for i,name in enumerate(labels):
            labelName_to_labelInd[name]=i
            labelInd_to_labelName[i]=name
    
        #save dictionnary
        with open(self.out_dir+"/labelName_to_labelInd_"+name_dict+".json", 'w') as f:
            json.dump(json.dumps(labelName_to_labelInd), f)
    
        with open(self.out_dir+"/labelInd_to_labelName_"+name_dict+".json", 'w') as f:
            json.dump(json.dumps(labelInd_to_labelName), f)
    
        return labelName_to_labelInd,labelInd_to_labelName
    
  

    def load_dico(self, path, key_int=False,print_me=False):
        """
        Load a dictionary from a JSON file.
    
        Parameters:
            -path (str): Path to the JSON file.
            -key_int (bool): If True, convert dictionary keys to integers. Otherwise, keep them as strings. Default is False.
            -print_me (bool): If True, print the loaded dictionary. Default is False.
    
        Returns:
            dict: Loaded dictionary.
        """
        
        
        
        
        with open(path) as f:
            dico_str = json.loads(json.load(f))
    
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
        
        dico={conv_key(k):v for k,v in dico_str.items()}
    
        if print_me:
            print(dico)
        
        return dico
       
    
    def get_annotation_information(self, annotation_file_name, original_sample_rate):
        
        '''
        Extract information from an annotation file.
    
        Parameters:
            - annotation_file_name: The name of the annotation file (without the file extension).
            - original_sample_rate: The original sample rate of the audio.
    
        Returns:
            - df_svl: A pandas DataFrame containing the extracted annotation information (Start, End, Label).
        '''


        # Process the .svl xml file
        xmldoc = minidom.parse(self.folder_annotation+'/'+annotation_file_name+'.svl')
        itemlist = xmldoc.getElementsByTagName('point')
        idlist = xmldoc.getElementsByTagName('model')

        start_time = []
        end_time = []
        labels = []
        audio_file_name = ''

        if (len(itemlist) > 0):

                
                print (annotation_file_name)
                
                # Iterate over each annotation in the .svl file (annotatation file)
                for s in itemlist:

                    # Get the starting seconds from the annotation file. Must be an integer
                    # so that the correct frame from the waveform can be extracted
                    start_seconds = float(s.attributes['frame'].value)/original_sample_rate
                    
                    # Get the label from the annotation file
                    label = str(s.attributes['label'].value)

                    # Set the default confidence to 10 (i.e. high confidence that
                    # the label is correct). Annotations that do not have the idea
                    # of 'confidence' are teated like normal annotations and it is
                    # assumed that the annotation is correct (by the annotator). 
                    label_confidence = 10

                    # Check if a confidence has been assigned
                    if ',' in label:

                        try:     
                            # Extract the raw label
                            lalel_string = label[:label.find(','):]

                            # Extract confidence value
                            label_confidence = int(label[label.find(',')+1:])

                            # Set the label to the raw label
                            label = lalel_string
                        except : 
                            raise TypeError("the label confidence number is missing on this file") 

                    # If a file has a blank label then skip this annotation
                    # to avoid mislabelling data
                    if label == '':
                        break

                    # Only considered cases where the labels are very confident
                    # 10 = very confident, 5 = medium, 1 = unsure this is represented
                    # as "SPECIES:10", "SPECIES:5" when annotating.
                    if label_confidence > 1 :
                        # Get the duration from the annotation file
                        annotation_duration_seconds = float(s.attributes['duration'].value)/original_sample_rate
                        start_time.append(start_seconds)
                        end_time.append(start_seconds+annotation_duration_seconds)
                        labels.append(label)

        df_svl = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})
        return df_svl 

    def get_meta_label(self, file_name_no_extension, database_ref, dictionary):
        '''
         Retrieve metadata and corresponding label for a given file.
         Requires a CSV file downloaded from Xenocanto
    
        Parameters:
            - file_name_no_extension: The name of the file without the file extension.
            - database_ref: The reference database containing metadata information obtained from Xenocanto.
            - dictionary: The dictionary mapping labels to numerical values.
    
        Returns:
            - meta: A numpy array containing the metadata (lat, lng, date, time, cnt).
            - y: The label corresponding to the file.
        ''' 
        
        
        
        #get back the ID number of the file
        xeno_id=file_name_no_extension.split('_')[5][2:]
            
            
        assert np.isin(xeno_id, database_ref['id']), "the id of the file {} is not contained in the reference database".format(file_name_no_extension)
                
        try: 
                meta=database_ref[database_ref['id']==int(xeno_id)][['lat','lng','date','time','cnt']].values[0]

                y=dictionary[database_ref[database_ref['id']==int(xeno_id)]['Scientifique_name'].values[0]]
        except : 
                raise TypeError("the metadata are not provided in the file") 
                
        return meta, y

    def getXY (self, audio_amplitudes, start_sec, annotation_duration_seconds, 
                   label, file_name_no_extension, database_ref, labelName_to_labelInd, verbose):
            '''
            Extract segments, corresponding metadata, and labels from an audio file based on annotations (svl file).

            Parameters:
                - audio_amplitudes: Amplitude values of the audio file.
                - start_sec: Starting time of the annotation in seconds.
                - annotation_duration_seconds: Duration of the annotation in seconds.
                - label: Label of the annotation.
                - file_name_no_extension: Name of the file without the file extension.
                - database_ref: Reference database containing metadata information from Xenocanto.
                - labelName_to_labelInd: Dictionary mapping label names to numerical values.
                - verbose: Boolean value indicating whether to print verbose output.

            Returns:
                - X_segments: List of audio segments.
                - X_meta_segments: List of corresponding metadata for each segment.
                -Y_labels: List of labels for each segment.
            '''
        
        
        
        
            X_segments = []
            X_meta_segments = []        
            Y_labels = []
            
            # Calculate how many segments can be extracted based on the duration of
            # the annotated duration. If the annotated duration is too short then
            # simply extract one segment. If the annotated duration is long enough
            # then multiple segments can be extracted.
            if annotation_duration_seconds-self.segment_duration < 0:
                segments_to_extract = 1
            else:
                segments_to_extract = annotation_duration_seconds-self.segment_duration+1
            
            if verbose:
                print ("segments_to_extract", segments_to_extract)


            for i in range (0, segments_to_extract):
                if verbose:
                    print ('Segment {} of {}'.format(i+1, segments_to_extract))
                    print ('*******************')
                
                # Set the correct location to start with.
                # The correct start is with respect to the location in time
                # in the audio file start+i*sample_rate
                start_data_observation = start_sec*self.downsample_rate+i*(self.downsample_rate)
                # The end location is based off the start
                end_data_observation = start_data_observation + (self.downsample_rate*self.segment_duration)
            
                # This case occurs when something is annotated towards the end of a file
                # and can result in a segment which is too short.
                if end_data_observation > len(audio_amplitudes):
                    continue
                
                # Extract the audio segment
                X_audio = audio_amplitudes[start_data_observation:end_data_observation]

                # Determine the actual time for the event in seconds
                start_time_seconds = start_sec + i
                                
                
                # Extract the metadata and label
                meta_location, label=self.get_meta_label(file_name_no_extension, database_ref, labelName_to_labelInd)
                
                X_segments.append(X_audio)
                X_meta_segments.append(meta_location)
                Y_labels.append(label)
                
            return X_segments, X_meta_segments, Y_labels
            
            
            
    def create_dataset(self, verbose):   
            '''
            Create the dataset from bird song audio files and corresponding annotations.
 
            Parameters:
                - verbose: Boolean value indicating whether to print verbose output.
 
            Returns:
                - X_calls: Array of audio segments.
                - X_meta: Array of corresponding metadata for each segment.
                - Y_calls: Array of labels for each segment.
            ''' 
            
            if verbose == True:
                print ('Annotations folder:',self.folder_annotation)
                print ('Audio path',self.folder)
            
            
        
            #create folder to store results
            if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
                    dico=False #dictionary need to be created
            else:
                    print("outdir already created")
                    dico=True  #dictionary has been already created previously
                    
            # Read database reference file
            database_ref=pd.read_csv(self.database_file,sep=';',encoding = "ISO-8859-1")  
            
            ##lists all audio files to be processed in the folder
            files=[f for f in listdir(self.folder)] 
            
            #load or create label dictionary       
            if dico==True:
                        labelName_to_labelInd=self.load_dico(self.out_dir+'/labelName_to_labelInd_22species.json',key_int=False,print_me=False)
            else: 
                        labels=np.unique([" ".join((x.split('_')[1],x.split('_')[2])) for x in files])
                        labelName_to_labelInd, labelInd_to_labelName =self.create_and_save_dictionnary(labels,'22species')
                        print('Dictionary has been created')
                         
            X_calls = []
            X_meta = []
            Y_calls = []  
            w=1 #counter
            

            for file in files: 
                
                print("files processed : ", w , 'on ', len(files))
                
                file_name_no_extension=file[:-4]
                print ('file_name_no_extension', file_name_no_extension)
                
                # Read audio file
                print ('Processing:',file_name_no_extension)
                audio_amps, audio_sample_rate = self.read_audio_file(self.folder+'/'+file)
                
                # Apply filtering and downsampling to the audio
                filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)
                amplitudes, sample_rate = self.downsample_file(filtered, audio_sample_rate, self.downsample_rate)
                 
                # Get annotations for the file
                annotation= self.get_annotation_information(file[:-4],audio_sample_rate)
                  
                for index, row in annotation.iterrows():

                    start_seconds = int(round(row['Start']))
                    end_seconds = int(round(row['End']))
                    label = row['Label']
                    annotation_duration_seconds = end_seconds - start_seconds

                    # # Extract audio segments, metadata, and labels
                    X_data, X_meta_, Y = self.getXY(amplitudes, start_seconds, 
                                                            annotation_duration_seconds, label, 
                                                            file_name_no_extension, database_ref, labelName_to_labelInd,  verbose)
                            
                    X_calls.extend(X_data)
                    X_meta.extend(X_meta_)
                    Y_calls.extend(Y)
                w+=1
            # convert to spectrogram if required
            if self.type_saved_data=='image':
                X_calls = self.convert_all_to_image(X_calls, sample_rate)

                  
            X_calls, X_meta, Y_calls = np.asarray(X_calls), np.asarray(X_meta), np.asarray(Y_calls)
              
            return X_calls, X_meta, Y_calls
                       
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
            
                
    def load_data_from_pickle(self, path, X, X_meta, Y,):
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
            infile = open(os.path.join(path, X+'.pkl'),'rb')
            X = pickle.load(infile)
            infile.close()
        
            infile = open(os.path.join(path, X_meta+'.pkl'),'rb')
            X_meta = pickle.load(infile)
            infile.close()
        
            infile = open(os.path.join(path, Y+'.pkl'),'rb')
            Y  = pickle.load(infile)
            infile.close()
        
            return X, X_meta, Y  
