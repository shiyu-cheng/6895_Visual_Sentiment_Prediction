# 6895_Visual_Sentiment_Prediction

## Project Overview
This project compares the performance of three fine-tuned models (ResNet50, ResNet101, and ResNext50) to predict sentiments from images on a dataset containing almost 10,000 images. ResNext50 performs the best among the three models, achieving an accuracy of almost 54\%. Our research subdivides the emotion categories and is the first to apply ResNext to the sentiment analysis of images. Results show that the bottleneck is the generation of the training dataset instead of the model.

## Dataset
All training and validation data are presented in the dataset folder.

## Model
The model folder contains various versions of code of our models. The trained models are in the subfolder models.

## UI
The UI is implemented by streamlit API. The UI folder includes the code, model, and images of our front-end deployment.

## Flickr
This is a depracated folder which contains our early-stage attempt data and code.
