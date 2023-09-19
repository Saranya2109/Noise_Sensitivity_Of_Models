# Linear Regression README

This repository contains Python code for simulating and analyzing linear regression models. The code demonstrates the process of fitting a linear regression model to random data, calculating various metrics, and visualizing the results.

## Table of Contents
- [Introduction](#introduction)
- [Simulating Random Data](#simulating-random-data)
- [Loss Function - Mean Squared Error](#loss-function---mean-squared-error)
- [R-squared](#r-squared)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable (output) and one or more independent variables (inputs). The relationship is expressed as:

y = w * x + b


In this code, I implement a simple linear regression model and apply it to randomly generated data. We then evaluate the model's performance using metrics such as Mean Squared Error (MSE) and R-squared.

## Simulating Random Data

I started by simulating random data points using the `get_data` function. The data consists of input `x` and output `y`, following a linear relationship with added noise. The data generation process includes setting a random seed for reproducibility.

## Loss Function - Mean Squared Error

The Mean Squared Error (MSE) is used as the loss function to evaluate the model's performance. MSE measures the average squared difference between the actual and predicted values:

MSE = (1/N) * Σ (y - ŷ)^2


## R-squared

I also calculated the R-squared value, which indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s). R-squared is defined as:

R^2 = 1 - Σ (y - ŷ)^2 / Σ (y - ȳ)^2


Where:
- `N` is the number of data points.
- `y` is the actual output.
- `ŷ` is the predicted output.
- `ȳ` is the mean of the actual output.

## Usage

To run this code, follow these steps:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Open a terminal or command prompt and navigate to the repository's directory.
4. Run the Python script using the following command:

- python linear_regression.py

The script will execute the linear regression simulations, compute the loss and R-squared values for different noise levels, and generate plots to visualize the results.

## Results
The results of the linear regression simulations are stored in a DataFrame and can be accessed in the LR_Results_loss.csv file. Additionally, various plots are generated to visualize the relationship between noise levels, loss, and R-squared values.

## License
This code is provided under the MIT License.

Feel free to use, modify, and distribute this code for educational and research purposes. If you find it helpful, please consider giving credit by referencing this repository.




