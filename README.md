# Dialect-Identification
A machine learning project on dialect identification.

## Introduction

- [x] This project is based on Gaussian Mixture Model.
- [x] The dialects which are used in the project are Bhojpuri, Maithili and Rajasthani, which belong to Hindi language.
- [x] The text data (which are recorded) are collect from the webite [News on AIR](https://newsonair.gov.in/).
- [x] Total 600 sample audios are used, 200 from each dialect.
- [x] The training set contains 80% of data and rest 20% is used for the testing purpose.
- [x] The accuracy achieved is 94.677%.
- [x] The features used are - MFCC, double delta.

## File Description
- [x] [FeaturesExtractor.py](https://github.com/pranav-pragyan/Dialect-Identification/blob/main/FeaturesExtractor.py) - This file is to extract the features from each audio file.
- [x] [ModelsTrainer.py](https://github.com/pranav-pragyan/Dialect-Identification/blob/main/ModelsTrainer.py) - This file is to train the model with training set and save the model.
- [x] [DialectIdentifier.py](https://github.com/pranav-pragyan/Dialect-Identification/blob/main/DialectIdentifier.py) - This file is to test the model created and output the accuracy.
