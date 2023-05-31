# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:06:17 2022

@author: ljeantet
"""



"""
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This class contains the main functions to convert audio windows into spectrograms.

The functions are similar to ones in Preprocessing_Xenocanto.py

"""


import librosa.display
import librosa
import numpy as np
from matplotlib import pyplot as plt





class convertisseur_spectro:
    def __init__(self, type_spec,                 
                 n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length ):
        
        self.type_spec=type_spec
    
        self.nyquist_rate = nyquist_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.win_length=win_length

    

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
        
    
        # 3 different input
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
        
    
        # 3 different input
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
        
        
    
        # 3 different input
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