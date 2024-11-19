# Automatic identification of the endangered Hawksbill sea turtle behavior from remote sensors using deep learning and cross-species transfer learning 



# Graphical Abstract

![abstract](https://github.com/jeantetlorene/TransferLearning_Vnet_Hawksbill/assets/105348746/0a9b08c9-5512-42c1-af70-33f1b69d755a)


### References 

[1] Brown et al. 2013 : Observing the unwatchable through acceleration logging of animal behavior. Animal Biotelemetry. doi: 10.1186/2050-3385-1-20.
[2] Nishizawa et al, 2013. Decision tree classification of behaviors in the nesting process of green turtles (Chelonia mydas) from tri-axial acceleration data. doi : 10.1007/s10164-013-0381-1
[3] Kumar et al. 2023. Human Activity Recognition ( HAR ) Using Deep Learning : Review , Methodologies , Progress and Future Research Directions. doi: 10.1007/s11831-023-09986-x. 
[4] Jeantet et al, 2021. Fully convolutional neural network : a solution to infer animal behaviours from multi-sensor data. doi: 10.1016/j.ecolmodel.2021.109555.
[5] Zhang et al. 2019. Human activity recognition based on motion sensor using u-net. doi: 10.1109/ACCESS.2019.2920969.

# Authors  

Lorène Jeantet [1,2,3], Kukhanya Zondo [1], Cyrielle Delvenne [4], Jordan Martin [4], Damien Chevallier [4] & Emmanuel Dufourq [1,2,3]

[1]: African Institute for Mathematical Sciences, South Africa \
[2]: African Institute for Mathematical Sciences, Research and Innovation Centre, Kigali, Rwanda \
[3]: Stellenbosch University, South Africa \
[4]: Unité de Recherche BOREA , MNHN, CNRS 8067, SU, IRD 207, UCN, UA, 
Station de Recherche Marine de Martinique, Quartier Degras, Petite Anse, 97217 Les Anses d'Arlet, Martinique, France

# Purpose of the project 

The objective of this project is to explore the application of transfer learning for automatically identifying behaviors using accelerometers and other sensors in wildlife studies. Deep learning remains underutilized in this field, and transfer learning has not been previously explored. Our proposal is to investigate transfer learning specifically for studying the behavior of the endangered hawksbill sea turtle. To achieve this, cameras combined with accelerometers were deployed by Damien Chevallier's team (CNRS France) on six individuals in Martinique. Moreover, this study is being conducted alongside research on green turtles, where we already possess annotated data and a deep learning model, a fully convolutional neural network, the V-net, trained on this population. 

Hence, considering the observed similarities in behavior, posture, and movement between green and hawksbill turtles, the primary aim of this study is to evaluate the feasibility of transferring insights obtained from a V-net model trained on green turtles to hawksbill turtles. 

Additionally, we explore the potential of transfer learning from other taxa, where data collection is more manageable. Additionally, we investigate the potential of transfer learning from other taxa, where data collection is more feasible. With this objective, we focused on humans, given the abundance of openly accessible datasets and the rapid and substantial advancements in Human Activity Recognition (HAR).


## Code  & Open source data 

To facilitate the understanding of the scripts and the re-use of the functions/models, we provide our code in the form of notebooks. Based on the description of the various tested scenarios in the associated article, you can find the following notebooks:

- Case2=Training_Vnet_Hawksbill: Notebook for training the V-net with random weight initialization on the hawksbill dataset.
- Case3=TransferLearning_Vnet_Hawksbill: Notebook for loading the pre-trained V-net on green turtles and fine-tuning it on hawksbill data. The weights of the model used in this notebook can be found in the folder Model_weights/Green_turtle
- Case4=TransferLearning_Unet_Human: Notebook for loading a pre-trained U-net model on the human dataset (IUC HAR) and fine-tuning it on hawksbill data. The model used in this notebook can be found in the folder Model_weights/Human

The labeled hawksbill dataset is freely accessible and can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.11402241 \
The green turtle dataset are available within the Dryad Digital Repository: https://doi.org/10.5061/dryad.hhmgqnkd9, and a preprocessed ready to use dataset will soon be available on Zenodo.

The human dataset is the publicly available <a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones">Intensive Care Unit (ICU) HAR dataset</a>. We employed an open-access U-net, proposed by <a href="https://ieeexplore.ieee.org/document/8731875"> Zhang et al. (2019)</a> and available on GitHub (https://github.com/zhangzhao156/Human-Activity-Recognition-Codes-Datasets/tree/master).

![transfer_learning_vs2](https://github.com/user-attachments/assets/2f07d58e-595c-4f2d-a687-b1d0de57cf0f)

The functions provided for training the V-net are from the GitHub repository developed by Dr Lorène Jeantet and Dr Vincent Vigon (https://github.com/jeantetlorene/Vnet_seaturtle_behavior), associated with the article: <a href="https://www.sciencedirect.com/science/article/abs/pii/S0304380021001253"> Fully Convolutional Neural Network: A solution to infer animal behaviours from multi-sensor data.</a> 
Please cite this article if you reuse the functions of these notebooks.


## Libraries 

python=3.10.12 \
tensorflow=2.15 \
pandas=2.0.3 \
numpy=1.25.2 \
matplotlib=3.8.4 \
scikit-learn==1.2.2

## Findings

Initially, we demonstrated that while behaviors may seem similar between green and hawksbill turtles, a model trained solely on green turtles could not accurately predict hawksbill behavior (Model-Hawksbill, F-score: 41.17), highlighting the necessity for species-specific model training. Conversely, utilizing a pre-trained model significantly improved predictions compared to a model trained from randomly initialized weights by 8% points (Model-Hawksbill, F1-score = 69.11 ; Model-Green_turtle.all layers fine-tuned , F1-score = 77.12). Additionally, the results suggested decreased variability in F1-scores when employing transfer learning. We achieved high F1-score by fine-tuning only the encoder, the initial layers of the model (Model-Green_turtle.Encoder fine-tuned , F1-score = 76.55). However, fine-tuning only the decoder resulted in lower performance (Model-Green_turtle.Decoder fine-tuned, F1-score = 53.21). While transfer learning in image classification often involves fine-tuning only the last layer, the softmax layer, our findings indicate that in our case with a fully convolutional network, fine-tuning only the softmax layer is not effective (Model-Green_turtle.Softmax fine-tuned, F1-score = 46.90). \
To assess the importance of the pre-trained model species choice, we tested transfer learning using a model trained on a significantly different species, humans. The results demonstrated 3.8% points increase in the F1-score compared to a model trained from randomly initialized weights (Model-Hawkbsill, F1-score = 69.11 ; Model-Human , F1-score = 72.87). This highlights that even when the behaviors of the pre-trained model species differ from the studied species, transfer learning enhances predictions with a similar training time.
![F1_score_article](https://github.com/user-attachments/assets/2964e30d-7d5f-4e9d-96da-4a6049e379b6)

