# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:49:53 2022

@author: ljeantet
"""
'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

File containing the parameters used in the study to process the audio files into spectogram
and the parameters of the architectures of the CNN used. 
'''



# I - Parameters of the preprocessing of the acoustic files

# Segmentation of the acoustic file
segment_duration=3
hop_chunck=0.5

# Downsample rate 
downsample_rate = 22050

# Parameters for the lowpass filter 
nyquist_rate =11025 # The Nyquist frequency of the signal
lowpass_cutoff = 10000 #The cutoff frequency of the filter


## Type of visual reprensation of the acoustic files we want to work with:
    ## 'spect' : The spectrogram representation of the audio
    ## 'melspect' : The mel-spectrogram representation of the audio.
    ## 'pcen' : The PCEN representation of the audio
type_spec='mel-spectro' 

# Parameters to build the visual representation of the acoustic files (spectrogram /melspectrogram/ pcen)
n_fft = 1024 # : length of the FFT window : Hann window length
hop_length=256 # number of samples between successive frames
n_mels=128  # number of Mel bands to generate
win_length=256 # Each frame of audio is windowed by window(). The window will be of length win_length and then padded with zeros to match n_fft.
f_min = 150 # lowest frequency (in Hz)
f_max=15000 # highest frequency (in Hz)


# II-Parameters of the models

#parameters of the base line model CNN
conv_layers = 1 #number of convolution layers to add after the first one
conv_filters = 8 #number of filters (same parameter for every convolution layer if conv_layers>0)
conv_kernel = 16 #size of the kernel (same parameter for every convolution layer if conv_layers>0)
max_pooling_size = 4  #integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
dropout_rate = 0.5  #Float between 0 and 1. Fraction of the input units to drop
fc_units_1 = 32 # nb of units of the first fully-connected layer



# Embedding parameters for the Multi-branch CNN (Case III)
"""To integrate the country names in a vector of chosen dimension, we started by assigning a unique numerical value between 0 and 50 (vocab_size) to each word present in our country list.  
Since some countries are composed of two words (e.g. South Africa, United Kingdom...), this results in a vector of size 2 (max_length), with a 0 in the second position for countries with only one word (e.g. Belgium, Venezuela, ..). 
We subsequently incorporated an embedding layer that mapped each value into an 8-dimensional transformed space (out_embedding), resulting in a vector of size [2, 8] for each country."""

vocab_size=50 #size of the vocabulary
max_length=2 ## max number of words in the country
out_embedding = 8 ##Dimension of the dense embedding

# Training
epoch=40
batch_size=32