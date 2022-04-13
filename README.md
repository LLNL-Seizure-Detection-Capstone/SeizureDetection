# Seizure Detection
The purpose of this project was to work under the direction of Lawrence Livermore National Labaratories to automate a seizure detection model.

## Introduction
In reviewing the current literature, we found that many research teams have investigated traditional machine learning architectures to accomplish a similar task; however, fewer teams had tested deep learning models. We built a custom dataset from a preprocessed subset of the CHB-MIT dataset to train an autoencoder that achieved 94\% accuracy on test data. We also began work on a transformer architecture that could be used to not only identify seizures recorded in the data but predict a seizure before its onset.

## Data
While testing the [CHB-MIT dataset](https://physionet.org/content/chbmit/1.0.0/), we found a preprocessed version of the database online. [The Karnataka State Akkamahadevi Women’s University](https://ieee-dataport.org/open-access/preprocessed-chb-mit-scalp-eeg-database#files) provided 68 minutes of both seizure and non-seizure data extracted from the .edf files and accessible in a single .csv file. We used the chbmit\_preprocessed\_data.csv file to train our model.

This helped us because the dataset was already balanced and we confirmed that when randomly sampling this for k folds validation it is close to a 50-50 split.
