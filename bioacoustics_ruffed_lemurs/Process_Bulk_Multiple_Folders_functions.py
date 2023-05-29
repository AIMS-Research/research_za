import glob, os
import ntpath

import random
from DataBank import *
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet101V2, ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import gc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import librosa.display
import librosa
from scipy import signal
import soundfile as sf
import datetime
import time

from yattag import Doc, indent
import ntpath


species_folder = 'E:/Lemurs_2022/SM1'
output_folder = 'E:/Lemurs_2022/Lemurs/Predictions_1' #new 9Nov 2021


meta = 'SH#{}#{}#Input channels: original spectrogram S, S**3, S**5.#'
INPUT_SHAPE = (128,151, 3)
save_results_folder = os.path.join(species_folder, 'Predictions_1')
number_classes = 2

# Spectrogram hyper-parameters
lowpass_cutoff = 4000 # Cutt off for low pass filter
downsample_rate = 9600 # Frequency to downsample to
nyquist_rate = 4800 # Nyquist rate (half of sampling rate)
segment_duration = 4 # how long should a segment be
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 500 # Spectrogram, minimum frequency for call
f_max = 9000 # Spectrogram, maximum frequency for call

# Name and location of the Tensorflow model
model_file_name = 'A7-1-712788074.hdf5'
model_filepath = '~/Lemurs/Weights/model=2023_03_14/'+model_file_name


if os.path.isdir(save_results_folder) == False:
    os.mkdir(save_results_folder)

# Reads an audio file and returns the amplitudes (audio data) and sample rate.
def read_audio_file(file_name):
    '''
    file_name: string, name of file including extension, e.g. "audio1.wav"

    '''

    # Read the amplitudes and sample rate
    audio_amps, audio_sample_rate = librosa.load(file_name, sr=None)

    return audio_amps, audio_sample_rate

# Designs a low-pass Butterworth filter given the cutoff frequency and Nyquist frequency.
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

# Applies a low-pass Butterworth filter to the input data.
def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Downsamples an audio file to a given new sample rate.
def downsample_file(amplitudes, original_sr, new_sample_rate):
    '''
    Downsample an audio file to a given new sample rate.
    amplitudes:
    original_sr:
    new_sample_rate:

    '''
    return librosa.resample(amplitudes, 
                            original_sr, 
                            new_sample_rate, 
                            res_type='kaiser_fast'), new_sample_rate

# Converts the amplitude values of an audio signal into a mel-spectrogram.
def convert_single_to_image(audio):
    '''
    Convert amplitude values into a mel-spectrogram.
    '''
    S = librosa.feature.melspectrogram(audio, n_fft=n_fft,hop_length=hop_length, 
                                       n_mels=n_mels, fmin=f_min, fmax=f_max)


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
    return np.stack([S1,S1**3,S1**5], axis=2)

# Converts a number of audio segments into their corresponding spectrograms.
def convert_all_to_image(segments):
    '''
    Convert a number of segments into their corresponding spectrograms.
    '''
    spectrograms = []
    for segment in segments:
        spectrograms.append(self.convert_single_to_image(segment))

    return np.array(spectrograms)

def add_extra_dim(spectrograms):
    '''
    Add an extra dimension to the data so that it matches
    the input requirement of Tensorflow.
    '''
    spectrograms = np.reshape(spectrograms, 
                              (spectrograms.shape[0],
                               spectrograms.shape[1],
                               spectrograms.shape[2],1))
    return spectrograms

#  Makes predictions using a trained model on a given test data.
def predict(model, X_test):
    ''' 
    Predict on one testing file and compute softmax output.
    
    '''
    
    # Get softmax predictions
    softmax_predictions = model.predict(X_test)
    
    return softmax_predictions

def load_model(model_filepath):
    
    model = tf.keras.models.load_model(model_filepath,compile=False)

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model

# Groups consecutive numbers from a list into sublists.
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

# Groups numbers in a list into consecutive ranges.
def group(L):
    L.sort()
    first = last = L[0]
    for n in L[1:]:
        if n - 1 == last: # Part of the group, bump the end
            last = n
        else: # Not part of the group, yield current group and start a new
            yield first, last
            first = last = n
    yield first, last # Yield the last group
    
# Converts a dataframe containing information about detected calls into a Sonic Visualiser Layer (SVL) XML format.  
def dataframe_to_svl(dataframe, sample_rate, length_audio_file_frames):

    doc, tag, text = Doc().tagtext()
    doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
    doc.asis('<!DOCTYPE sonic-visualiser>')

    with tag('sv'):
        with tag('data'):
            
            model_string = '<model id="1" name="" sampleRate="{}" start="0" end="{}" type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" subtype="box" minimum="0" maximum="{}" units="Hz" />'.format(sample_rate, 
                                                                        length_audio_file_frames,
                                                                        sample_rate/2)
            doc.asis(model_string)
            
            with tag('dataset', id='0', dimensions='2'):

                # Read dataframe or other data structure and add the values here
                # These are added as "point" elements, for example:
                # '<point frame="15360" value="3136.87" duration="1724416" extent="2139.22" label="Cape Robin" />'
                for index, row in dataframe.iterrows():

                    point  = '<point frame="{}" value="{}" duration="{}" extent="{}" label="{}" />'.format(
                        int(int(row['start(sec)'])*sample_rate), 
                        int(row['low(freq)']),
                        int((int(row['end(sec)'])- int(row['start(sec)']))*sample_rate), 
                        int(row['high(freq)']),
                        row['label'])
                    
                    # add the point
                    doc.asis(point)
        with tag('display'):
            
            display_string = '<layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
            doc.asis(display_string)

    result = indent(
        doc.getvalue(),
        indentation = ' '*2,
        newline = '\r\n'
    )

    return result


def calculate_vocalisation_time(file_name, detected_time):
    
    print ('calculate_vocalisation_time')
    print (type(file_name))
    print (file_name)
    print (type(detected_time))
    print (detected_time)
    print (type(int(detected_time)))
    
    date_time = file_name[file_name.find('_')+1:]
    date = date_time[:date_time.find('_')]
    time = date_time[date_time.find('_')+1:]
    now = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(time[0:2]), int(time[2:4]), int(time[4:6]))
    difference1 = datetime.timedelta(seconds=detected_time)
    new_time = now + difference1
    
    return new_time.strftime("%Y%m%d_%H%M%S")

def save_detected_calls(groupped_detection, amplitudes, file_name, sample_rate, sub_folder_name):
    
    if len(groupped_detection) == 0:
        print ('No predictions to save.')
        return

    for counter, group in enumerate(groupped_detection):
        print (counter)
        print (group)
        #Added 23 Nov 2021
        if len(group) > 4:
            start_index = group[0]-2 # Add 2 seconds before
            end_index = group[-1]+2 # Add 2 seconds after

            print (start_index)
            print (end_index)

            print ('Saving...')
            folder = save_results_folder+'/'
            time_stamp = calculate_vocalisation_time(file_name, int(start_index))
            sf.write(folder+sub_folder_name+'/'+file_name+'_'+str(counter)+'_detected_'+str(time_stamp)+'.wav', amplitudes[sample_rate*start_index:sample_rate*end_index], sample_rate)

        print ('')

# Processes a single audio file by applying filtering, downsampling, 
# converting to spectrograms, making predictions, and saving the results.
def process_one_file(file_name,species_folder, model, sub_folder_name):
    
    print ('Reading audio file...')
    print(species_folder +file_name)
    audio_amps, original_sample_rate = read_audio_file(species_folder +file_name)
    
    print ('Done reading file')
    print ('Filtering...')

    # Low pass filter
    filtered = butter_lowpass_filter(audio_amps, lowpass_cutoff, nyquist_rate)
    
    print ('Done filtering amplitudes')

    print ('Downsampling...')
    # Downsample
    amplitudes, sample_rate = downsample_file(filtered, original_sample_rate, downsample_rate)
    
    print ('Done downsampling')
    
    del filtered
    
    start_values = np.arange(0, len(audio_amps)/original_sample_rate - segment_duration).astype(np.int)
    end_values = np.arange(segment_duration, len(audio_amps)/original_sample_rate).astype(np.int)
    
    amplitudes_to_predict = []

    print ('Converting amplitudes to spectrograms...')
    
    for i in range (len(start_values)):
        s = start_values[i]
        e = end_values[i]

        #print (s, e) 
        S = convert_single_to_image(amplitudes[s*sample_rate:e*sample_rate])
        amplitudes_to_predict.append(S)
    
    len_audio_amps=len(audio_amps)
    del audio_amps
    
    print ('Predicting...')

    amplitudes_to_predict = np.asarray(amplitudes_to_predict)
    print (amplitudes_to_predict.shape)
    
    softmax_predictions = predict(model, amplitudes_to_predict)
    
    del amplitudes_to_predict
    
    print ('Done predicting')
    
    binary_predictions = []
    prediction_seconds = []
    for index, softmax_values in enumerate(softmax_predictions):
        if softmax_values[1] < 0.5:
            binary_predictions.append('Noise')
            prediction_seconds.append(0)
        else:
            binary_predictions.append('Lemur')
            prediction_seconds.append(1)
    
        
    print ('Grouping calls together')
    # Group the detections together
    groupped_detection = group_consecutives(np.where(np.asarray(prediction_seconds) == 1)[0])
    
    print (groupped_detection)
    print ('len:', len(groupped_detection[0]))
    
    if len(groupped_detection[0])> 0:
        print ('Saving the detected clips')
        
        # Create a dataframe to store each prediction
        predictions = []
        for pred in groupped_detection:

            if len(pred) >= 2:
                #print (pred)
                for predicted_second in pred:
                    # Update the set of all the predicted calls
                    predictions.append(predicted_second)
        
        predictions.sort()

        # Only process if there are consecutive groups
        if len(predictions) > 0:
            predicted_groups = list(group(predictions))
            
            print ('Predicted')

            # Create a dataframe to store each prediction
            df_values = []
            for pred_values in predicted_groups:
                df_values.append([pred_values[0], pred_values[1]+segment_duration, 400, 1100, 'predicted'])
            df_preds = pd.DataFrame(df_values, columns=[['start(sec)','end(sec)','low(freq)','high(freq)','label']])
            print(df_preds.shape)
            # Create a .svl outpupt file
            xml = dataframe_to_svl(df_preds, original_sample_rate, len_audio_amps)

            # Write the .svl file
            #text_file = open(species_folder+'/Model_Output/'+file_name_no_extension+"_"+self.model_type+".svl", "w")
            text_file = open('{}_final_L_prediction.svl'.format(save_results_folder+'/'+sub_folder_name+'/'+file_name), "w")
            n = text_file.write(xml)
            text_file.close()
        
    else:
        print ('No detected calls to save.')
    del amplitudes
    
    print ('Done')

# Main function that processes all the audio files in 
# the specified folder by calling the process_one_file() function for each file.
def process_files():
    
    global species_folder

    #list_of_files_not_pred = glob.glob(os.path.join(species_folder, 'NotPredicted/*.wav'))
    list_of_files_not_pred = glob.glob(os.path.join(species_folder, '**/*.wav'), recursive=True)
    
    # Maybe ignore the prediction folder if the code has been run before? (write "ignore" if first run)
    filteredResults = [r for r in list_of_files_not_pred if not "Predictions" in r]
    list_of_files_not_pred = filteredResults
    
    if len(list_of_files_not_pred) == 0:
        print ('No new files need predicting.')
        return
    else:
        print ('{} new file(s) need to be processed.'.format(len(list_of_files_not_pred)))
        model = load_model(model_filepath)

    for file in list_of_files_not_pred:
        
        #sub_folder = file[file[:file.rfind('/')].rfind('/')+1:file.rfind('/')]
        sub_folder=file.split('\\')[1]
        sub_folder
        
        species_folder_sub = species_folder+'/'+sub_folder+'/'
        
        file_name = ntpath.basename(file)
        
        print('Processing file:', file_name)
        
        if os.path.isdir(save_results_folder+'/'+sub_folder) == False:
            os.mkdir(save_results_folder+'/'+sub_folder)
        
        process_one_file(file_name,species_folder_sub, model, sub_folder)

    del model
    
start_time = time.time()
    
process_files()

end = time.time()
print(start_time-end)