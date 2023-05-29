import os.path
from os import path
import datetime
import shutil
import random
import numpy as np
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import randint
from tensorflow.keras.utils import to_categorical

import pickle

class DataBank:
    ''' Manage the data. Allow for data to be sampled in a 
    reproducible manner.
    '''
    
    def __init__(self, species_folder, test_fraction, augmentation_amount, type,
                 X_file_name, Y_file_name, save_results_folder):
        
        self.species_folder = species_folder
        self.seed = self.initialise_seed()
        self.test_fraction = test_fraction
        self.label_encoder = LabelEncoder()
        self.augmentation_amount = augmentation_amount
        self.type = type
        self.X_name=X_file_name
        self.Y_name=Y_file_name
        self.save_results_folder=save_results_folder

    def initialise_seed(self):
        ''' Initialise a seed to a random integer.
        '''
        return (random.randint(2, 1147483648))

    def set_seed(self, seed):
        self.seed = seed
                
    def get_seed(self):
        ''' Get the seed value
        '''
        return self.seed
    
    def generate_new_seed(self):
        ''' Creat a new seed
        '''
        self.seed = self.initialise_seed()
        
    def __load_picked_data(self):
        '''
        Load all of the spectrograms from a pickle file
        
        '''

        # Check which type of pre-processed input to use
        # duplicated spectrograms, different hop values
        # or powered values.
        if self.type == 'hop':
            X_data_name = 'X-hop.pkl'
            Y_data_name = 'Y-hop.pkl'

        if self.type == 'dup':
            X_data_name = 'X-dup.pkl'
            Y_data_name = 'Y-dup.pkl'

        if self.type == 'pow':
            X_data_name = 'X_balanced-pow.pkl'
            Y_data_name = 'Y_balanced-pow.pkl'

        if self.type == 'pow2':
            X_data_name = 'X-pow2.pkl'
            Y_data_name = 'Y-pow2.pkl'
        
        if self.type == 'large_pow':
            X_data_name = 'X_large_pow.pkl'
            Y_data_name = 'Y_large_pow.pkl'
        
        if path.exists(os.path.join(self.species_folder, 'Saved_Data', self.X_name+"-pow.pkl")) == False:
            print ('Pickled Data X does not exist.\nCreate it first.')
            raise Exception('Pickled Data X does not exist.')
            
        if path.exists(os.path.join(self.species_folder, 'Saved_Data', self.Y_name+"-pow.pkl")) == False:
            print ('Pickled Data Y does not exist.')
            raise Exception('Pickled Data Y does not exist.\nCreate it first.')
        
        infile = open(os.path.join(self.species_folder, 'Saved_Data', self.X_name+"-pow.pkl"),'rb')
        X = pickle.load(infile)
        infile.close()
        
        infile = open(os.path.join(self.species_folder, 'Saved_Data', self.Y_name+"-pow.pkl"),'rb')
        Y = pickle.load(infile)
        infile.close()

        return X, Y
    
    def create_new_folder(self):
        
        now = datetime.datetime.now()
        today=now.strftime("%Y_%m_%d")
        dir_out=self.save_results_folder+"/model="+today
        
       
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
            print("we clear the directory:",dir_out)
        else:
            print("we create the directory:",dir_out)
    
        """création des dossiers """
        os.makedirs(dir_out)
        return dir_out
        
    def __randomly_sample_data(self, amount_to_sample, X, Y, positive_class_label):
        ''' Randomly sample a given amount of examples from the positive class
        without replacement. All the examples from the negative class remain 
        constant. E.g. randomly get 20 examples of Cape Robin-Chat calls
        will return X and Y values which contain all the original noise
        and only 20 examples of Cape Robin-Chats.
        '''
        
        # Get all indices of the examples which match the positive class
        species_indices = np.where(Y == positive_class_label)[0]
        
        # Get all indices of the examples which match the negative class
        non_species_indices = np.where(Y != positive_class_label)[0]
        
        # If the number of elements to randomly sample is greater than
        # the amount of data, then sample with replacement.
        if amount_to_sample > len(species_indices):
            
            # Randomly select (with replacement)
            randomly_selected_idx = list(np.random.choice(list(species_indices), amount_to_sample, replace=True))
        else:
            
            # Randomly select (without replacement)
            randomly_selected_idx = random.sample(list(species_indices), amount_to_sample)
        
        # Get the spectrograms of the species of interest
        X_augmented, Y_augmented = self.__augment_with_time_shift(X[randomly_selected_idx], positive_class_label)

        print (X_augmented.shape)
        print (Y_augmented.shape)

        # Get the spectrograms and labels of background noise
        X_background = X[list(non_species_indices)]
        Y_background = Y[list(non_species_indices)]

        print (X_background.shape)
        print (Y_background.shape)

        #X_augmented.extend(X_background)
        #X_augmented.extend(X_background)

        # Append all the negative class examples as that remains constant
        #randomly_selected_idx.extend(list(non_species_indices))
        
        # Assign the new X and Y values (i.e. all the original noise
        # and a subset of the calls of interest)
        #X = X[randomly_selected_idx]
        #Y = Y[randomly_selected_idx]

        X = np.concatenate((X_augmented, X_background))
        Y = np.concatenate((Y_augmented, Y_background))
        
        print (X.shape)
        print (Y.shape)
        
        # Shuffling needed
        X, Y = self.__shuffle_data(X, Y)
        
        # Return the randomly selected examples
        return X, Y
        
    def __shuffle_data(self, X, Y):
        ''' Shuffle the X, Y paired data
        '''
        X, Y = shuffle(X, Y, random_state=self.seed)
        
        return X,Y

    def __create_traintest_split(self, X, Y):
        ''' Split the data into training and testing
        '''
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
            test_size=self.test_fraction, random_state=self.seed, 
            shuffle=True)
        
        return X_train, X_test, Y_train, Y_test


    def __augment_with_time_shift(self, X_data, positive_class_label):
        ''' Augment the spectrograms by shifting them to some random
        time steps to the right.
        '''

        augmented_spectrograms = []
        augmented_labels = []

        # Iterate over all the spectrograms for the species of
        # interest.
        for X in X_data:

            # Create a number of augmentated spectrograms
            for i in range (0, self.augmentation_amount):

                # Randomly select amount to shift by
                random_shift = randint(1, X.shape[1]-1)

                # Time shift
                shifted_spectrogram = np.roll(X, random_shift, axis=1)

                # Append the augmented segments
                augmented_spectrograms.append(shifted_spectrogram)

                # Append the class labels
                augmented_labels.append(positive_class_label)

        # Return the augmented spectograms and labels
        return np.asarray(augmented_spectrograms), np.asarray(augmented_labels)
        
    def get_data(self, amount_to_sample, positive_class_label):
        ''' Randomly sample a pre-defined number of X,Y pairs
        while keeping the examples from the background noise
        constant and randomly sampling from the positive class.
        '''
        # Set the seed
        random.seed(self.get_seed())
        
        # Read in the X and Y data
        X, Y = self.__load_picked_data()
        print (X.shape)
        print (Y.shape)

        # Add fix here
        print (np.unique(Y))

        # Small fix when using 3 classes and main class is PTW
        fixed_Y = []
        for label in Y:
            new_label = label.replace("CRC", "NOISE")
            fixed_Y.append(new_label)

        fixed_Y = np.asarray(fixed_Y)
        # Add fix here
        print (np.unique(Y))
        print (np.unique(fixed_Y))

        # Create train test split
        X_train, X_test, Y_train, Y_test = self.__create_traintest_split(X, fixed_Y)
        print ('?')
        print (X_train.shape)
        print (Y_train.shape)
        print (X_test.shape)
        print (Y_test.shape)

        # Randomly sample from the data
        # removed on 23 November as a test.
        #X_train, Y_train = self.__randomly_sample_data(amount_to_sample, X_train, Y_train, positive_class_label)

        print (np.unique(Y_train))
        print(len(np.unique(Y_train)))

        # Transform the Y labels into one-hot encoded labels
        Y_transformer = self.label_encoder.fit(Y_train)
        Y_train = Y_transformer.transform(Y_train)
        Y_test = Y_transformer.transform(Y_test)

        print ('---')
        print(len(np.unique(Y_train)))
        print(len(np.unique(Y_test)))
        
        Y_train = to_categorical(Y_train, num_classes = len(np.unique(Y_train)))
        Y_test = to_categorical(Y_test, num_classes = len(np.unique(Y_test)))

        del X, Y


        # Return the spectrograms, labels and the seed used
        # The full test set is returned.
        # The sampled training set is returned (i.e. the training set
        # will only contained `amount_to_sample` number of examples of
        # the species of interest)
        return X_train, X_test, Y_train, Y_test, self.seed
