# IMDb Movie Rating Prediction

This project focuses on predicting IMDb movie ratings based on features such as genre, director, actor names, and duration. It utilizes a Gradient Boosting Regressor model for predicting ratings.

## Dataset

The dataset used for this project is the "IMDb Movies India" dataset, which can be found [here](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies). The dataset includes information about various movies, including features such as genre, director, actor names, duration, and IMDb ratings.

## Exploratory Data Analysis (EDA)

Before building the model, we performed exploratory data analysis to gain insights into the dataset. Key visualizations include:

- Distribution of movie durations
- Relationship between movie duration and IMDb ratings
- Categorical counts of genres and directors


## Model Building

The model is built using a Gradient Boosting Regressor, a popular ensemble learning technique. Features like actor names and director are combined into a single text feature, and movie duration is treated as a numerical feature.

## Model Evaluation

The performance of the model is evaluated using the Mean Squared Error (MSE) metric. MSE measures the average squared difference between predicted and actual ratings. Lower MSE values indicate better predictive performance.

## Making Predictions

Users can interact with the model by providing actor names, director name, and movie duration as input. The model then predicts the IMDb rating for the input movie.

## Usage

To use the model, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/CodeRreaper69/CODSOFT_DATA_SCIENCE
```
2. Navigate to the project directory:

 ```bash
cd CODSOFT_DATA_SCIENCE/TASK_1
```
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the prediction script:

```bash
python movie_prediction.py
or
python3 movie_prediction.py
```

5. Enter actor names, director name, and movie duration when prompted and get the predicted movie rating


