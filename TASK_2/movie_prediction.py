import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler





df = pd.read_csv("IMDb Movies India.csv", encoding='latin-1')#dataframe object
#print(df["Age"].max())
#function for data insights
def data_insights(df):
    
    #df = pd.read_csv(d)#dataset from kaggle
    #printing first 5 rows for it
    print("FIRST 5 ROWS ARE-")
    print(df.head())
    print("\n")

    #printing the info about the dataset
    print("Information about the dataset-")
    print(df.info())
    print("\n")


    #taking important statistical insights of the dataset
    print("Some statistics of the dataset -")
    print(df.describe())
    print("\n")


    #printing the structure of the dataset
    print("ROWS,COLUMNS")
    print(df.shape)
    print("\n")

    print("columns are -")
    print(df.columns)

#for data visualization
def data_vis(df,field1,field2,field3,field4):
    #creating excel reading object
    #df = pd.read_csv(df)#dataset from kaggle
    print(df[field1].value_counts())
    print(df[field2].value_counts())
    #sns.countplot(x=field1, hue=field2, data=df)
    fig, axes = plt.subplots(1,2)

    # Plotting first subplot
    axes[0].set_title(f"{field1} vs {field2}")
    sns.countplot(x=field1, hue=field2, data=df, ax=axes[0])
    
    # Plotting second subplot
    axes[1].set_title(f"{field3} vs {field4}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[1])
    
    # enabling the plot
    #plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
    plt.close()



#print("Before dropping the null values")
#data_insights(df)

columns_to_use = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Rating']
df = df[columns_to_use]
df = df.dropna()

print("After dropping the null values")

data_insights(df)

# Combining text features into a single column
df['TextFeatures'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Actor 1'] + ' ' + df['Actor 2'] + ' ' + df['Actor 3']

# Converting 'Duration' to integers
df['Duration'] = df['Duration'].str.replace(r'\D', '', regex=True).astype('int')

data_vis(df,"Duration","Rating","Duration","Genre")

X = df[['TextFeatures', 'Duration']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Preprocessing the text features (Genre, Director, Actors)
text_features = ['TextFeatures']
text_transformer = Pipeline(steps=[
    ('vectorizer', CountVectorizer())
])

# Preprocessing for duration feature
duration_features = ['Duration']
duration_transformer = Pipeline(steps=[
    ('duration_scaler', StandardScaler())
])

# Combining transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'TextFeatures'),
        ('duration', duration_transformer, duration_features)
    ])

# Creating a Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)])

pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)


print("\n \n")
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')





print("\n \n")
print("GIVE ACTOR AND DIRECTOR NAMES WITH MOVIE DURATIONS AND GET THE RATING OF THE MOVIE")
print("\n")

a = 0
while True:

    # User Input
    actor_name = input("Enter Actor Name (separate multiple actors with spaces): ")
    director_name = input("Enter Director Name: ")
    duration = int(input("Enter Movie Duration in minutes: "))

    # Combining user input into a single text feature
    user_text_features = f"{actor_name} {director_name}"

    # Make a prediction for the user input
    user_input = pd.DataFrame({'TextFeatures': [user_text_features], 'Duration': [duration]})
    predicted_rating = pipeline.predict(user_input)[0]

    print(f"\nPredicted Movie Rating: {predicted_rating}")
    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        a = True
print("\n")



















