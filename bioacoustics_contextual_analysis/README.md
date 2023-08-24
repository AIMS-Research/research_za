# Improving deep learning acoustic classifiers with contextual information for wildlife monitoring

# Graphical Abstract 
![abstract image](https://github.com/AIMS-Research/research_za/assets/105348746/7581228f-667f-4647-8166-22faaea76d5a)

## Purpose of the project

Just as a human would use contextual information to identify species calls from acoustic recordings, one unexplored way to improve deep learning classifier in bioacoustics is to provide the algorithm with contextual meta-data, such as time and location. In this study, we exlpored different methods to provide location information to a bird song classifier.

We answer the question: given two bird songs, as shown below, can machine learning learn to differentiate between the songs by using spectrograms and additional information (such as geographical information).

![Figure1](https://github.com/AIMS-Research/research_za/assets/15357701/ff83d68d-9c7b-462d-9960-67057217873b)

Link to the research article: https://www.sciencedirect.com/science/article/pii/S1574954123002856

Keywords: bioacoustics, deep learning, passive acoustic monitoring, species identification, birds, Hainan gibbons

# Authors 
Lorène Jeantet [1,2,3] & Emmanuel Dufourq [1,2,3]

[1]: African Institute for Mathematical Sciences, South Africa  
[2]: Stellenbosch University, Department of Applied Mathematics, South Africa  
[3]: National Institute for Theoretical and Computational Sciences, South Africa  

Machine Learning for Ecology research group


## Open source data

<a href="https://doi.org/10.5281/zenodo.7828148 ">Our training data is open source and can be accessed here. </a> We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding labelled text files (.data). 

The dataset contains songs of 22 bird species from 5 families and genera differents. The recordings were downloaded from the Xeno-canto database in .wav format and each recording was manually annotated by labelling the start and stop time for every vocalisation occurrence using Sonic Visualiser. In total, database contained 6537 occurrences of bird songs of various length from 967 file recordings. A precise description of the distribution by species and country can be found in the associated article. See below an example of an annotated file.

![Figure2_exampleannotation](https://github.com/AIMS-Research/research_za/assets/15357701/66a11441-e225-428b-a4b2-ea1ab1d2ebab)

We also used vocalisations from Hainan gibbons in a second case study. <a href="https://zenodo.org/record/7997739">The testing set and multi-branch model is available here.</a> 

![gibbons_calls](https://github.com/AIMS-Research/research_za/assets/15357701/2fa9e027-ea1e-4d15-ad10-8db341064f9c)

# Findings

We developed a multi-branch convolutional neural network (CNN) that leverages spectrograms as the primary input and incorporates spatial metadata as a secondary input to classify 22 distinct bird songs. We compared this approach to a baseline model using only spectrogram input. Additionally, we trained a geographical prior neural network to estimate the likelihood of a species occurring at a specific location. The output of this network was integrated with the baseline CNN. In a separate case study, we employed temporal and spectrogram data as inputs to a multi-branch CNN for the detection of Hainan gibbon calls, the rarest primate in the world. Our results demonstrate that augmenting metadata significantly enhances the performance of the bird song classifier, with the greatest improvement achieved using the geographical prior model (F1-score of 87.78% versus 61.02% for the baseline model). The multi-branch CNNs also proved to be efficient (achieving F1-scores of 76.87% and 78.77%) and simpler to implement than the geographical prior. Our model is shown below.

![Figure3_BaseLine_Geographical_1](https://github.com/AIMS-Research/research_za/assets/15357701/7cfcf1fe-6c10-42f4-89f7-dcd9abe89f69)


In the second case study, leveraging metadata with the multi-branch CNN resulted in a 63% reduction in false positives (detecting 94% of calls) and a 19% increase in gibbon detection. 

## Libraries

Tested on Python 3.11
- soundfile==0.12.1
- librosa==0.10.1
- numpy==1.23.5
- yattag==1.15.1
- pandas==2.0.1
- scipy==1.10.1
- scikit-learn==1.2.2
- matplotlib==3.7.1
- tensorflow==2.11.0

## Code description 

### Organisation of the folders

The main folder contains the scripts necessary to pre-process the dataset and to train the Baseline model as well as the multi-branch CNN. The scripts to train the Geographical prior can be found in the dedicated folder "Geographical_prior". 

The scripts are written to save the pre-processed data into the "ou"' folder and the trained models, along with their outputs, into the "Models_out" folder. When training a model, a new folder associated with that experiment will be created in "Models_out" with the date of the experiment and the model name as the folder's name.

The "out" folder contains dictionaries used in the article to map between an index number and the name of the species or the country. Functions to open the dictionnaries or generate them can be found in the "Preprocessing_Xenocanto" class ("Preprocessing_Xenocanto.py"). 

For the Greographical prior, the data necessary to train the model can be found in the "Data" folder. Similarly to the main folder, when training the model, a new folder associated with that experiment will be created in 'Models_out' with the date of the experiment and the model name as the folder's name.

### Data pre-processing
![preprocessing](https://github.com/AIMS-Research/research_za/assets/105348746/3af49704-ccfc-4524-a8a6-c6719f5650d4)
![Data_augmentation](https://github.com/AIMS-Research/research_za/assets/105348746/84a5b55a-898f-45e7-b577-24aefcf356fe)

### Training of the Baseline Model and Multi-branch CNN 
![Training](https://github.com/AIMS-Research/research_za/assets/105348746/2c698a87-63f7-4ed4-b10e-eb55696bfa44)

### Training and Application of the Geographical Prior 
The codes related to the training process of the Geographical prior can be found in the folder Geographical_prior. 

![train_geo_prior](https://github.com/AIMS-Research/research_za/assets/105348746/afd13062-15dc-4ef1-aa56-985f91a0dc80)
![eval_geo_prior](https://github.com/AIMS-Research/research_za/assets/105348746/36e2c3b7-08f3-4d99-8003-1e2f3b841454)


