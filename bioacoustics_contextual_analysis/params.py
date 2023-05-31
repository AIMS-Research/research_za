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



# Parameters of the spectrograms
n_fft = 1024 # Hann window length
hop_length=256 # Sepctrogram hop size
n_mels=128
nyquist_rate =11025
lowpass_cutoff = 10000
win_length=256
f_min = 150 # minimum frequency for call
f_max=15000

#type of spectogram to use ()
type_spec='mel-spectro' #betwenn 'spectro','mel-specto','pcen'
downsample_rate = 22050


# II-Parameters of the models

#parameters of the base line model CNN
conv_layers = 1
fc_layers = 2
max_pooling_size = 4 
dropout_rate = 0.5
conv_filters = 8
conv_kernel = 16
fc_units_1 = 32
fc_units_2 = 32

# Embedding layer for the Multi-branch CNN (Case III)
vocab_size=50
max_length=2

# Training
epoch=40
batch_size=32