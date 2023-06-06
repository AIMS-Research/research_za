# Empowering Deep Learning Acoustic Classifiers with Human-like Ability to Utilize Contextual Information for Wildlife Monitoring

## Purpose of the project

Just as a human would use contextual information to identify species calls from acoustic recordings, one unexplored way to improve deep learning classifier in bioacoustics is to provide the algorithm with contextual meta-data, such as time and location. In this study, we exlpored different methods to provide location information to a bird song classifier.

We answer the question: given two bird songs, as shown below, can machine learning learn to differentiate between the songs by using spectrograms and additional information (such as geographical information).

![Figure1](https://github.com/AIMS-Research/research_za/assets/15357701/ff83d68d-9c7b-462d-9960-67057217873b)

Link to the research article: [to appear soon]

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


