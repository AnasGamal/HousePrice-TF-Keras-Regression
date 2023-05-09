# Week 1 Assignment: Housing Prices

This repository contains the Week 1 assignment for the "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning" course on Coursera. The goal of this project is to build a neural network that predicts the price of a house based on a simple formula.

## Problem Description

Imagine that house pricing is as easy as:

A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k, etc.

The objective is to create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k, etc. To improve the performance of the neural network, the house prices are scaled down during training, and the predictions are made in the 'hundreds of thousands'.

## Implementation

The project demonstrates the implementation of a basic regression model using TensorFlow and Keras. The model has been trained on a small dataset containing the number of bedrooms and their corresponding house costs.

The model architecture consists of a single dense layer with one unit and uses Stochastic Gradient Descent as the optimizer, along with the Mean Squared Error loss function. After training, the model is saved into a file with the extension `.h5`. This allows for the trained model to be easily loaded for future predictions without requiring retraining. The model is only trained if the saved file does not exist, ensuring efficient use of resources.

The main script, `house_price_predictor.py`, interacts with the user by prompting them to input the number of bedrooms for their desired house. Based on this input, the saved model is used to predict the cost of the house. The predicted cost is displayed in hundreds of thousands, providing a user-friendly output that is easy to interpret.

## Prerequisites

To run the project, you'll need the following:

- Python 3.6 or later
- TensorFlow 2.x
- Numpy

## Installation

1. Clone this repository to your local machine.
2. Install the required packages using the following command:
`pip install -r requirements.txt`
3. Run the script `house_price_predictor.py` to train the model and make predictions.

## Usage

The main script `house_price_predictor.py` can be executed in a terminal or an IDE. The script prompts the user to input the number of bedrooms, and then it predicts the cost of the house based on the provided input. The predicted cost is displayed in hundreds of thousands.

## Course Reference

This project is part of the ["Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning" course on Coursera](https://www.coursera.org/learn/introduction-tensorflow).

## License

This project is open-source and available under the MIT License.