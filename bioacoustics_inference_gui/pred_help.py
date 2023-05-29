from xml.dom import minidom
import math
import pandas as pd
import glob, os
import librosa.display
import librosa
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow.keras import backend, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet101V2, ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import gc
import tkinter as tk


from yattag import Doc, indent
import ntpath

class Prediction:
    
    def __init__(self, lowpass_cutoff, 
                 downsample_rate, nyquist_rate, segment_duration, 
                 n_fft, hop_length, n_mels, f_min, f_max):
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        #self.current_task = current_task
        #self.running_check = running_check
        self.weights_name = ''

    def setweights_name(self, weights_name):
        self.weights_name = weights_name
    
    #def still_running (self):
    #    print('checking', self.running_check )
    #    return self.running_check 

    #def stop_running(self):
    #    print('stop')
    #    self.current_task.config(text = 'Current Task: Stopping...')
    #    self.running_check = False

    #def reset(self):
    #    self.running_check = True



    def load_model(self):
    
        model = tf.keras.models.load_model(self.weights_name )
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        return model

    def group_consecutives(self, vals, step=1):
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

    def group(self,L):
        L.sort()
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last: # Part of the group, bump the end
                last = n
            else: # Not part of the group, yield current group and start a new
                yield first, last
                first = last = n
        yield first, last # Yield the last group

    def dataframe_to_svl(self, dataframe, sample_rate, length_audio_file_frames):

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

    def calculate_vocalisation_time(self, file_name, detected_time):
        
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

        

        
    def read_audio_file(self, file_name):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"

        '''

        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(file_name, sr=None)

        return audio_amps, audio_sample_rate

    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
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

    def convert_single_to_image(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_fft,hop_length=self.hop_length, 
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
        return np.stack([S1,S1**3,S1**5], axis=2)

    def convert_all_to_image(self, segments):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)

    def add_extra_dim(self, spectrograms):
        '''
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        '''
        spectrograms = np.reshape(spectrograms, 
                                  (spectrograms.shape[0],
                                   spectrograms.shape[1],
                                   spectrograms.shape[2],1))
        return spectrograms

    def predict(self, model, X_test):
        ''' 
        Predict on one testing file and compute softmax output.
        
        '''
        
        # Get softmax predictions
        softmax_predictions = model.predict(X_test, batch_size=16)
        
        return softmax_predictions
    def process_one_file(self, file_name, model, folder_output_name):
        
        print ('Reading audio file...')
        #self.current_task.delete("1.0", tk.END)
        #self.current_task.insert(tk.INSERT, 'Current Task: Reading audio'+str(file_name))
        #self.current_task.update()
        audio_amps, original_sample_rate = self.read_audio_file(file_name)
        
        print ('Done reading file')
        print ('Filtering...')

        # Low pass filter
        ##self.current_task.delete("1.0", tk.END)
        #self.current_task.insert(tk.INSERT, 'Current Task: filtering')
        #self.current_task.update()
        filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff , self.nyquist_rate)
        
        print ('Done filtering amplitudes')

        print ('Downsampling...')
        # Downsample
        #self.current_task.delete("1.0", tk.END)
        #self.current_task.insert(tk.INSERT, 'Current Task: downsampling')
        #self.current_task.update()
        amplitudes, sample_rate = self.downsample_file(filtered, original_sample_rate, self.downsample_rate)
        
        print ('Done downsampling')
        
        del filtered
        
        start_values = np.arange(0, len(audio_amps)/original_sample_rate - self.segment_duration).astype(np.int)
        end_values = np.arange(self.segment_duration, len(audio_amps)/original_sample_rate).astype(np.int)
        
        amplitudes_to_predict = []

        print ('Converting amplitudes to spectrograms...')
        #self.current_task.delete("1.0", tk.END)
        #self.current_task.insert(tk.INSERT, 'Current Task: creating spectrograms')
        #self.current_task.update()
        
        for i in range (len(start_values)):
            s = start_values[i]
            e = end_values[i]

            #print (s, e) 
            S = self.convert_single_to_image(amplitudes[s*sample_rate:e*sample_rate])
            amplitudes_to_predict.append(S)

        len_audio_amps=len(audio_amps)
        del audio_amps
        
        print ('Predicting...')
        #self.current_task.delete("1.0", tk.END)
        #self.current_task.insert(tk.INSERT, 'Current Task: predicting')
        #self.current_task.update()

        amplitudes_to_predict = np.asarray(amplitudes_to_predict)
        print (amplitudes_to_predict.shape)
        
        softmax_predictions = self.predict(model, amplitudes_to_predict)
        
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
        groupped_detection = self.group_consecutives(np.where(np.asarray(prediction_seconds) == 1)[0])
        
        print (groupped_detection)
        print ('len:', len(groupped_detection[0]))
        
        if len(groupped_detection[0])> 0:
            print ('Saving the detected clips')
            # Save the groupped detections to .wav output clips
            #save_detected_calls(groupped_detection, amplitudes, file_name, sample_rate, sub_folder_name)
            # Save the groupped predictions to .svl file to visualize them with Sonic Visualizer
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
                predicted_groups = list(self.group(predictions))
                
                print ('Predicted')

                # Create a dataframe to store each prediction
                df_values = []
                for pred_values in predicted_groups:
                    df_values.append([pred_values[0], pred_values[1]+self.segment_duration, 400, 1100, 'predicted'])
                df_preds = pd.DataFrame(df_values, columns=[['start(sec)','end(sec)','low(freq)','high(freq)','label']])
                print(df_preds.shape)
                # Create a .svl outpupt file
                xml = self.dataframe_to_svl(df_preds, original_sample_rate, len_audio_amps)

                # Write the .svl file
                #text_file = open(species_folder+'/Model_Output/'+file_name_no_extension+"_"+self.model_type+".svl", "w")
                text_file = open('{}/{}_prediction.svl'.format(folder_output_name, os.path.basename(file_name)), "w")
                n = text_file.write(xml)
                text_file.close()
          
        else:
            print ('No detected calls to save.')
        del amplitudes
        
        print ('Done')
