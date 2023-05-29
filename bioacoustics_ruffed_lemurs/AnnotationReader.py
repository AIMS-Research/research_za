import pandas as pd
import os
import librosa
from xml.dom import minidom

# Reads a .svl file which contains the start/stop time for each annotated even
# along with the class label.

class AnnotationReader:
    def __init__(self, annotation_file_name, path, file_type, audio_extension):
        
        self.annotation_file_name = annotation_file_name
        self.path = path
        self.file_type = file_type
        self.audio_extension = audio_extension
        
    # This function reads an audio file and returns the amplitudes and sample rate of the audio.
    def read_audio_file(self, file_name, species_folder):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"

        '''
        # Get the path to the file
        audio_folder = os.path.join(species_folder, 'Audio',file_name)

        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)

        return audio_amps, audio_sample_rate

    # This reads the .svl file generated from Sonic Visualiser after an audio file
    # has been manually annoated. These .svl files are essentially .XML files and thus
    # each row in the file can be read to extract the start/stop time of each event.
    # Each annotated event also has a label and this is extracted too. The function
    # returns a Pandas dataframe which has the start/stop time for each annotated
    # even along with the label.
    def get_annotation_information(self):

        if self.file_type == 'svl':
            # Process the .svl xml file
            xmldoc = minidom.parse(self.path+'/Annotations/'+self.annotation_file_name+'.svl')
            itemlist = xmldoc.getElementsByTagName('point')
            idlist = xmldoc.getElementsByTagName('model')

            start_time = []
            end_time = []
            labels = []
            audio_file_name = ''

            if (len(itemlist) > 0):

                file_name_no_extension = self.annotation_file_name
                print (file_name_no_extension)
                audio_amps, original_sample_rate = self.read_audio_file(file_name_no_extension+self.audio_extension ,self.path)

                # Iterate over each annotation in the .svl file (annotatation file)
                for s in itemlist:

                    # Get the starting seconds from the annotation file. Must be an integer
                    # so that the correct frame from the waveform can be extracted
                    start_seconds = int(int(s.attributes['frame'].value)/original_sample_rate)

                    

                    # Get the label from the annotation file
                    label = str(s.attributes['label'].value)

                    # Set the default confidence to 10 (i.e. high confidence that
                    # the label is correct). Annotations that do not have the idea
                    # of 'confidence' are teated like normal annotations and it is
                    # assumed that the annotation is correct (by the annotator). 
                    label_confidence = 10

                    # Check if a confidence has been assigned
                    if ',' in label:

                        # Extract the raw label
                        lalel_string = label[:label.find(','):]

                        # Extract confidence value
                        label_confidence = int(label[label.find(',')+1:])

                        # Set the label to the raw label
                        label = lalel_string

                    # Only considered cases where the labels are very confident
                    # 10 = very confident, 5 = medium, 1 = unsure this is represented
                    # as "SPECIES:10", "SPECIES:5" when annotating.
                    if label_confidence == 10:
                        # Get the duration from the annotation file
                        annotation_duration_seconds = int(int(s.attributes['duration'].value)/original_sample_rate)
                        start_time.append(start_seconds)
                        end_time.append(start_seconds+annotation_duration_seconds)
                        labels.append(label)

            df_svl_gibbons = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})
            return df_svl_gibbons, file_name_no_extension+'.wav'


        if self.file_type == 'raven_caovitgibbons':
            df = pd.read_csv(os.path.join(self.path+'/Annotations/', self.annotation_file_name), sep='\t')

            start_time = []
            end_time = []
            labels = []
            audio_file_name = ''

            for index, row in df.iterrows():
                start_time.append(row['Begin Time (s)'])
                end_time.append(row['End Time (s)'])
                labels.append(row['Label'])

            df_raven_gibbons = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})

            return df_raven_gibbons

        if self.file_type == 'raven_hyenas':
            df = pd.read_csv(os.path.join(self.path+'/Annotations/', self.annotation_file_name), sep='\t')

            start_time = []
            end_time = []
            labels = []
            audio_file_name = ''

            for index, row in df.iterrows():
                start_time.append(row['Begin Time (s)'])
                end_time.append(row['End Time (s)'])
                labels.append(row['Species'])

            df_raven_gibbons = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})

            return df_raven_gibbons
    
    # This function determines the audio location based on the annotation file name. 
    # If the annotation file name contains a hyphen ('-'), it extracts the part before 
    # the last hyphen and converts any other hyphens to forward slashes ('/'). 
    # Otherwise, it returns an empty string.
    def get_audio_location(self):
        
        if '-' in self.annotation_file_name:
            return '/'.join(self.annotation_file_name[:self.annotation_file_name.rfind('-')].split('-'))+'/'
        else:
            return ''