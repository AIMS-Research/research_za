# Graphical User Interface for Bioacoustics Inference

## Purpose of the project

To development a graphical user interface (GUI) which can easily be used to predict presence/absence of a vocalising species on audio files. This was originally developed for the detection of black-and-white ruffed lemur vocalisations.

Keywords: conservation, ecoacoustics, machine learning, CNNs, bioacoustics, passive acoustic monitoring

## Authors

Emmanuel Dufourq [1],[2],[3], Lorene Jeantet [1],[2],[3]

[1]: African Institute for Mathematical Sciences, South Africa  
[2]: Stellenbosch University, Department of Applied Mathematics, South Africa  
[3]: National Institute for Theoretical and Computational Sciences, South Africa  

<img src="https://github.com/AIMS-Research/research_za/assets/15357701/ecf37866-e80b-4d90-9a5b-b60b096c3ffc" width=50%>                                                                        

## YouTube demo tutorial
We provide <a href="https://youtu.be/PAyO8dK7lCk">a video which demonstrates the steps needed to execute the software </a>. This should be accompanied with the user manual below.

## User manual

A user manual is <a href="https://docs.google.com/document/d/1x86dZ3vVTSvxcWJnuc9KpMUELWBVNyMVtk85Imd2pi0/edit?usp=sharing
">provided here. </a> 


## Open source data

<a href="https://doi.org/10.5281/zenodo.7956064">Our training, testing and model is open source and can be accessed here. </a> We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding labelled text files (.data). Tensorflow model weights are provided. This dataset has 56 testing audio files (roughly 38 hours, 8.7GB) and 246 training files.

DOI for data: 10.5281/zenodo.7956064

The dataset was labelled as a binary classification problem, either vocalisations which contained roar shrieks (labelled as 'roar') or did not (labelled as 'no-roar'). The roars varied in amplitude, duration and environmental noise. See below for some examples of parts of audio files which were manually annotated.

A 48 second roar shriek event:
<img src="https://drive.google.com/uc?export=view&id=1gNiBJTgJOeipOp417VTIQHNyFsdpGl3e" width="900">

An example of another animal vocalisation:
<img src="https://drive.google.com/uc?export=view&id=1Q0GuQSPPJKP5TueefvSv_hqDkMscfCK_" width="900">

### Libraries:

Tested on Python 3.9.16

soundfile==0.12.1

librosa==0.8.1

numpy==1.23.5

yattag==1.15.1

pandas==2.0.0

scipy==1.10.1

matplotlib==3.7.1

tensorflow==2.12.0

