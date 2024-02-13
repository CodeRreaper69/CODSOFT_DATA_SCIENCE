#importing all the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#creating excel reading object
df = pd.read_csv("D:/machine_learning/CS_1/Titanic-Dataset.csv") #dataset location
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
def data_vis(df,field1,field2):
    #creating excel reading object
    #df = pd.read_csv(df)#dataset from kaggle
    print(df[field1].value_counts())
    print(df[field2].value_counts())
    sns.countplot(x=field1, hue=field2, data=df)
    plt.show() 

#data_insights(df)
#data_vis(df,'Survived','Pclass')

#dropping null values
df = df.dropna(subset=['Age', 'Sex', 'Survived'])

#print(df.info())
#data_insights(df)
#data_vis(df,'Survived','Sex')






#Encoding categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

#Creating age groups
bins = [1,10,20,30,40,50,60,70,80]
labels = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Select features and target variable
X = df[['AgeGroup', 'Sex']]
y = df['Survived']


#creating dummies
X = pd.get_dummies(X, columns=['AgeGroup'], drop_first=True)

#training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#Creating a logistic regression model
model = LogisticRegression(random_state=42)

#Training the model
model.fit(X_train, y_train)

#Making predictions on the test set
y_pred = model.predict(X_test)

#Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

#prediction making and taking input from the user
a = 0
while True:
    age = int(input("ENTER THE AGE OF THE INDIVIDUAL: "))
    s = input("ENTER SEX OF THE INDIVIDUAL: (m/f/M/F):")

    if s in ['M','m']:
        sex = 'male'
    else:
        sex = 'female'


    new_data = pd.DataFrame({'Age': [age], 'Sex': [sex]})

    # Encode categorical variables
    new_data['Sex'] = label_encoder.transform(new_data['Sex'])

    # Create age group
    new_data['AgeGroup'] = pd.cut(new_data['Age'], bins=bins, labels=labels, right=False)

    # One-hot encode the 'AgeGroup' column
    new_data = pd.get_dummies(new_data[['AgeGroup', 'Sex']], columns=['AgeGroup'], drop_first=True)

    # Make predictions
    prediction = model.predict(new_data)

    # Output the prediction
    if prediction[0]==0:
        print(f"The Individual with age {age} and gender {sex} would not have survived")
    else:
        print(f"The Individual with age {age} and gender {sex} would have survived")

    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        a = True
        
    



















