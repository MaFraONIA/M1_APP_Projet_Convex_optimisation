# Projet_Convex_Optimisation

This project is centered around the experimentation of logistic regression with a Lasso penalty, specifically utilizing the Iterative Soft-Thresholding Algorithm (ISTA) for optimization. It aims to compare the effectiveness and efficiency of the ISTA algorithm against classic gradient descent and other machine learning models like Random Forest, Support Vector Machines (SVM), and Neural Networks using the breast cancer Wisconsin dataset from sklearn.

## Overview

The project is structured into several Python scripts, each handling a different aspect of the machine learning experimentation process:
- `data_loader.py` for loading and preprocessing the dataset.
- `ista_algorithm.py` and `gradient_descent.py` for implementing the logistic regression models with the ISTA algorithm and classic gradient descent, respectively.
- `model_training.py` for training and evaluating all models.

The project utilizes Python and libraries such as NumPy, scikit-learn, and Matplotlib for data manipulation, model training, and results visualization.

## Features

- Implementation of logistic regression with Lasso penalty using ISTA.
- Performance comparison of ISTA with classic gradient descent and other models.
- Evaluation of models based on accuracy, ROC curve, AUC, and other metrics.
- Visualization of model performances for easy comparison.

## Getting started

### Requirements

- Python 3.8 or above
- NumPy
- scikit-learn
- Matplotlib
- Pandas

### Quickstart

1. Ensure all requirements are installed by running `pip install numpy scikit-learn matplotlib pandas`.
2. Load and preprocess the dataset by running `python data_loader.py`.
3. Train and evaluate the models using `python model_training.py`.
4. Visualize the comparison of model performances.

### License

Copyright (c) 2024.