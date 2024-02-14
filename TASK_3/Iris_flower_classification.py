import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report





df = pd.read_csv("IRIS.csv")#dataframe object
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

    print(df.columns.max())
    print(df.columns.min())

    print("Null values:")
    print(df.isna().sum())
    
#for data visualization
def data_vis(df,field1,field2,field3,field4,field5,field6,field7,field8):
    #creating excel reading object
    #df = pd.read_csv(df)#dataset from kaggle
    #print(df[field1].value_counts())
    #print(df[field2].value_counts())
    #sns.countplot(x=field1, hue=field2, data=df)
    fig, axes = plt.subplots(2,2)

    # Plotting first subplot
    axes[0,0].set_title(f"{field1} vs {field2}")
    sns.countplot(x=field1, hue=field2, data=df, ax=axes[0,0])
    
    # Plotting second subplot
    axes[0,1].set_title(f"{field3} vs {field4}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[0,1])

    # Plotting third subplot
    axes[1,0].set_title(f"{field5} vs {field6}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[1,0])

    # Plotting fourth subplot
    axes[1,1].set_title(f"{field7} vs {field8}")
    sns.countplot(x=field3, hue=field4, data=df, ax=axes[1,1])
    
    # enabling the plot
    plt.tight_layout()  # Adjusting layout to prevent overlapping
    plt.show()
    plt.close()


data_insights(df)
data_vis(df,'species','sepal_length','species','sepal_width','species','petal_length','species','petal_width')


#model training a d predicitions


#Extracting feauture and target variables
X = df.drop('species', axis=1)
y = df['species']

#splitting training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state = 15)

#standardising feautured values
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#training a K-nearest neighbour classification
k = 3 #taking value of this k
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_trained_scaled, y_train)


#making predictions on the test set
y_pred = knn_model.predict(X_test_scaled)


#the accuracy of this training
accuracy =  accuracy_score(y_test,y_pred)
print("\n")
print(f"ACCURACY: {accuracy*100}%")
print("\n")


#classificition summary and report
print("CLASSIFICATION SUMMARY: ")
print(classification_report(y_test,y_pred))


print("\n \n")

print("ENTER THE MEASUREMENTS AND GET THE SPECIES OF THE FLOWER")

print("\n")

#user inputs and predictions
a = 0
while True: 
    sepal_length = float(input("Enter sepal length (4.3 to 7.9) : "))
    sepal_width = float(input("Enter sepal width (2.0 to 4.4) : "))
    petal_length = float(input("Enter petal length (1.0 to 6.9) : "))
    petal_width = float(input("Enter petal width (0.1 to 2.50) :"))

    # Creating a DataFrame with the help of user input and feature names
    user_input_df = pd.DataFrame(
        {'sepal_length': [sepal_length],
         'sepal_width': [sepal_width],
         'petal_length': [petal_length],
         'petal_width': [petal_width]}
    )

    
    # Standardizing the user input using the same scaler used for training
    user_input_scaled = scaler.transform(user_input_df)

    # Making a prediction using the trained model
    predicted_species = knn_model.predict(user_input_scaled)

    # Displaying the predicted species
    print(f"The predicted species is: {predicted_species[0]}")

    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        pass


print("\n")































