# Seizure Detection
The purpose of this project was to work under the direction of Lawrence Livermore National Laboratories to automate a seizure detection model.

## Introduction
In reviewing the current literature, we found that many research teams have investigated traditional machine learning architectures to accomplish a similar task; however, fewer teams had tested deep learning models. We built a custom dataset from a preprocessed subset of the CHB-MIT dataset to train an autoencoder that achieved 94\% accuracy on test data. We also began work on a transformer architecture that could be used to not only identify seizures recorded in the data but predict a seizure before its onset.

## Data
While testing the [CHB-MIT dataset](https://physionet.org/content/chbmit/1.0.0/), we found a preprocessed version of the database online. [The Karnataka State Akkamahadevi Women’s University](https://ieee-dataport.org/open-access/preprocessed-chb-mit-scalp-eeg-database#files) provided 68 minutes of both seizure and non-seizure data extracted from the .edf files and accessible in a single .csv file. We used the chbmit\_preprocessed\_data.csv file to train our model.

This helped us because the dataset was already balanced and we confirmed that when randomly sampling this for k folds validation it is close to a 50-50 split.

## Model
After preprocessing our data successfully we went and created a Deep Learning Auto Encoder Classifier model using the PyTorch framework. We treated each 2 second sample of EEG data as a 2D array using the axis values of time and EEG node for each measurement. We used a Convolutional Neural Network (CNN) as the Auto Encoder of this data. We found the architecture and were inspired by the work of [Abdelhameed and Bayoumi](https://www.frontiersin.org/articles/10.3389/fncom.2021.650050/full) but we could not find published code for this paper.

We tuned the hyper parameters of the model deciding that the best hyper parameters. The following are the best that we found.

* Optimizer: Adam
* Learning Rate: .001
* Batch Size: 32
* Activation: ReLu
* Dropout Rate: 20\%

## Acknowledgements
Reflecting on our capstone experience, we would like to take the time to acknowledge the many people and organizations that allowed this project to be completed in a timely manner. Thank you to the team who partially preprocessed the dataset that allowed for easy access to the CHB-MIT dataset to quickly get a training dataset for our model. In addition, we would like to thank the researchers who provided a good sample architecture for us to us to build our autoencoder using PyTorch. Thank you to Lawrence Livermore National Laboratory for providing us with a mentor, Cindy Gonzales, and for the amazing mentorship she provided our team. Furthermore, we would like to thank our advisor provided by Brigham Young University, Dr. Giraud-Carrier for his hands-on mentorship with us. Without all of the people listed above this project would not have been possible.
