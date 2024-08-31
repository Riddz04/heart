import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
heart = pd.read_csv('heart.csv')

# X->model data y->output of that particular row
X = heart.iloc[:, :13]
y = heart.iloc[:, -1]

# Apply standard scaler to the data
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

# Splitting the train and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scale,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# Random forest classifier
rf = RandomForestClassifier(criterion='entropy', max_depth=1)
rf.fit(X_train, y_train)

# Creating a class for the user
class HeartDiseaseDiagnosis:
    def __init__(self, rf, scaler):
        # Initialize the RandomForestClassifier and StandardScaler
        self.rf = rf
        self.scaler = scaler
        self.y_pred = None

    def Diagnosis(self, X_test):
        # Since X_test is already scaled, we do not need to scale it again
        # Make predictions
        self.y_pred = self.rf.predict(X_test)
        # Print the accuracy score
        print('accuracy_score:', accuracy_score(self.y_pred, y_test))

    def getData(self):
        # Print the predictions
        print(self.y_pred)

# Example of how to use the class
diagnosis = HeartDiseaseDiagnosis(rf, scaler)
diagnosis.Diagnosis(X_test)
diagnosis.getData()
