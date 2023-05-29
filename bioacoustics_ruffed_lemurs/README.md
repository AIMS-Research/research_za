# An integrated passive acoustic monitoring and deep learning pipeline applied to black-and-white ruffed lemurs (Varecia variegata) in Ranomafana National Park,Madagascar

## Purpose of the project

Fieldwork was conducted at Mangevo (21.3833S, 47.4667E), an isolated and undisturbed forest location within Ranomafana National Park (RNP), located in southeastern Madagascar, during the period of May to July 2019. To facilitate passive acoustic monitoring, we deployed a total of two SongMeter SM4 devices (manufactured by Wildlife Acoustics) and two Swift units (provided by the Cornell Yang Center for Conservation Bioacoustics). The placement of these recorders was strategically chosen within the central regions of known subgroups, ensuring a minimum distance of 300 meters between each device. The SongMeter devices operated at a sampling rate of 48 kHz, while the Swift units operated at 32 kHz, respectively, enabling comprehensive audio data collection throughout the study period.

Link to the research article: [to appear soon]

Keywords: conservation, ecoacoustics, machine learning, strepsirrhine, CNNs, bioacoustics, passive acoustic monitoring

## Authors

Carly H. Batist [1],[2],[9], Emmanuel Dufourq [3],[4],[5], Lorene Jeantet [3],[4],[5], Mendrika N.Razafindraibe [6], Francois Randriamanantena [7], and Andrea L. Baden [1],[8], [9]

[1]: The Graduate Center of the City University of New York, Department of Anthropology, New York, USA  
[2]: Rainforest Connection (RFCx), Katy, USA  
[3]: African Institute for Mathematical Sciences, South Africa  
[4]: Stellenbosch University, Department of Applied Mathematics, South Africa  
[5]: National Institute for Theoretical and Computational Sciences, South Africa  
[6]: University of Antananarivo, Department of Animal Biology, Antananarivo, Madagascar  
[7]: Centre ValBio, Ranomafana, Madagascar  
[8]: Hunter College of the City University of New York, Department of Anthropology, New York, USA  
[9]: New York Consortium in Evolutionary Primatology (NYCEP); New York, NY, USA

## Demo

A quick 3 minute demo is available here on <a href="https://colab.research.google.com/drive/1G_zicIHNTrBJuiXJYsKqRMWktazdX3vx?usp=sharing">Google Colab demo</a>. The script applies the model to one audio file and generated predictions. The audio file and predictions can be loaded into Sonic Visualiser to that the predictions can easily be verified as shown:

![image](https://drive.google.com/uc?export=view&id=11ZA6GRCQcCJD6f7kFc3_EybuWzdiOUQ1)

A very faint black-and-white ruffed lemur vocalisation can be seen in the spectrogram above. The model is able to detect calls that were recorded far away from the microphone.

## Open source data

<a href="https://doi.org/10.5281/zenodo.7956064">Our training, testing and model is open source and can be accessed here. </a> We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding labelled text files (.data). Tensorflow model weights are provided. This dataset has 56 testing audio files (roughly 38 hours, 8.7GB) and 246 training files.

DOI for data: 10.5281/zenodo.7956064

The dataset was labelled as a binary classification problem, either vocalisations which contained roar shrieks (labelled as 'roar') or did not (labelled as 'no-roar'). The roars varied in amplitude, duration and environmental noise. See below for some examples of parts of audio files which were manually annotated.

A 48 second roar shriek event:
<img src="https://drive.google.com/uc?export=view&id=1gNiBJTgJOeipOp417VTIQHNyFsdpGl3e" width="900">

An example of another animal vocalisation:
<img src="https://drive.google.com/uc?export=view&id=1Q0GuQSPPJKP5TueefvSv_hqDkMscfCK_" width="900">

Roar shriek and rain:
<img src="https://drive.google.com/uc?export=view&id=1fQybOjJ4_WnlxUwMmCbB4H_qcqOUC9Sj" width="900">

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

## Code description

Below we provide a high level overview of the Python scripts.

### Data pre-processing

![Preprocessing](https://github.com/jeantetlorene/Lemurs_TransferLearning/assets/15357701/8066c073-15d5-42f3-aeab-5bd6eddb3482)

Since the model was pre-trained on ImageNet, it expects a 3-channel input. We followed the approach described in Dufourq, E., Batist, C., Foquet, R. and Durbach, I., 2022. Passive acoustic monitoring of animal populations with transfer learning. Ecological Informatics, 70. Below is an example of a spectrogram, denoted as S, which contains Hainan gibbon vocalisations and  was used to create two additional spectrograms. The additional spectrograms were created by taking the exponent of each pixel within S. In the first case each pixel was raised to the power of 3, and in the second, to the power of 5. Since the spectrograms were normalised between 0 and 1, the values in the new spectrograms do not exceed 1.

![1-s2 0-S1574954122001388-gr2](https://github.com/jeantetlorene/Lemurs_TransferLearning/assets/15357701/6ea7febe-5477-465f-8f3c-a0f689a04d59)

### Training

![Training](https://github.com/jeantetlorene/Lemurs_TransferLearning/assets/15357701/3dcbd0d5-54ac-4bf0-80ad-9cd5615c69f1)

### Inference and predicting on new data

![Prediction](https://github.com/jeantetlorene/Lemurs_TransferLearning/assets/15357701/47a9f5d1-f0d4-4e4f-8911-8b7a543135f7)


