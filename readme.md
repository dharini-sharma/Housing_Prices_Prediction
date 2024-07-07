# Housing Price Prediction Project

This project aims to predict housing prices using a neural network model built with TensorFlow and Keras. The dataset used is from the California Housing Prices dataset, which includes various features such as location, demographics, and economic data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Files](#files)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to develop a machine learning model that can predict housing prices based on given features. The model uses data from California Housing Prices.The model is implemented using TensorFlow's Keras API, with regularization techniques to prevent overfitting.

## Dataset

The dataset used (`housing.csv`) contains the following columns:
- `longitude`: A numeric value for the longitude of the location.
- `latitude`: A numeric value for the latitude of the location.
- `housing_median_age`: Median age of housing units in the block.
- `total_rooms`: Total number of rooms in the block.
- `total_bedrooms`: Total number of bedrooms in the block.
- `population`: Total number of people residing in the block.
- `households`: Total number of households, a group of people residing in a home unit.
- `median_income`: Median income of households in the block.
- `median_house_value`: Median house value in units of currency.

## Preprocessing

Before training the neural network model:
- Rows with NULL values from column `total_bedrooms` were removed from the dataset to ensure data integrity before training the neural network model.
- The `ocean_proximity` column was dropped as it was categorical.
- Features (`x`) and target (`y`) were scaled using `StandardScaler`.
- The data was split into training, validation, and testing sets. The training set comprises 60% of the data, while both the validation and testing sets consist of 20% each of the dataset.

## Model Architecture

The neural network architecture consists of:
- Input layer
- Two hidden layers with ReLU activation and L2 regularization
- Output layer (single neuron for predicting house prices)

## Training

The model was trained for 100 epochs with a batch size of 32. The Adam optimizer and Mean Squared Error loss function were used.

## Evaluation

The model's performance was evaluated using the validation set (`xval`, `yval`) during training. Metrics such as Mean Squared Error and Mean Absolute Error were used to assess performance.

## Files

- `housing.csv`: Dataset used for training and evaluation.
- `housing_price_prediction.ipynb`: Jupyter notebook containing the Python code for the project.
- `README.md`: This file providing an overview of the project.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn