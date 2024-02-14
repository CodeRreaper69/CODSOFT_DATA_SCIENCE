# Iris Flower Species Classification

## Overview

This repository contains a Python script for training a machine learning model to classify Iris flowers into different species based on their sepal and petal measurements. The dataset used for this task is the Iris Flower Dataset, available on Kaggle [here](https://www.kaggle.com/datasets/arshid/iris-flower-dataset).

## Content

- **File**: `Iris_flower_classification.py`
  - The main Python script for loading the Iris dataset, exploring insights, visualizing data, training a k-Nearest Neighbors (k-NN) classification model, and making predictions based on user input.

- **Folder**: `TASK_3`
  - The folder containing the Python script (`Iris_flower_classification.py`).

## Process

1. **Data Insights and Visualization:**
   - The script begins by loading the Iris dataset (`IRIS.csv`) using Pandas and exploring various insights about the data using functions like `data_insights` and `data_vis`. This helps in understanding the structure and characteristics of the dataset.

2. **Model Training:**
   - Features and target variables are extracted from the dataset, and the data is split into training and testing sets. The features are standardized using `StandardScaler`.
   - A k-Nearest Neighbors (k-NN) classifier is trained on the standardized training set.

3. **Model Evaluation:**
   - The accuracy of the trained model is evaluated on the testing set, and a classification summary is displayed.

4. **User Input and Predictions:**
   - The script then prompts the user to enter sepal and petal measurements, and the trained model predicts the species based on the user input.

## How to Run

1. **Install Dependencies:**
   - Ensure you have Python installed on your system.
   - Install required libraries using: `pip install numpy pandas seaborn matplotlib scikit-learn`.

2. **Clone the Repository:**
   - Clone this repository to your local machine: `git clone https://github.com/CodeRreaper69/CODSOFT_DATA_SCIENCE`.

3. **Navigate to the Folder:**
   - Open a terminal and go to the `TASK_3` folder: `cd CODSOFT_DATA_SCIENCE/TASK_3`.

4. **Run the Script:**
   - Execute the script: `Iris_flower_classification.p`.

5. **User Input:**
   - Follow the prompts to enter sepal and petal measurements for predictions.

## Dataset Source

- Kaggle: [Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

