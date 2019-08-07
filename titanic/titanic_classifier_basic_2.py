"""

Titanic: Machine Learning from Disaster
Date started: 7th August 2019
Author: Naji Aziz

"""

# Import basic modules
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, rc={'figure.figsize':(20,15)})



# Pull data
df_train = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/train.csv")
df_test = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/test.csv")

# Brief exploration of data
df_train.info() # Age, Cabin and Embarked have missing values
df_test.info() # Age, Fare and Cabin have missing values

df_train.head()
df_train.tail()

df_train.describe()

sns.distplot(df_train["Fare"].dropna()) # Distribution exploration

# Quick and dirty cleaning job on train set
df_train_clean = df_train.drop(["Cabin"], axis = 1) # Drop Cabin column

median_train_age = df_train_clean.Age.median() # Calc median Age from train set (naive)
df_train_clean.Age = df_train_clean.Age.fillna(median_train_age) # Replace train Ages with median

df_train_clean = df_train_clean.dropna() # Drop all missing values from train dataset

df_train_clean.info() # 889 clean observations remain
df_train_clean.tail()
df_train_clean.describe()

# Quick and dirty cleaning job on test set
df_test_clean = df_test.drop(["Cabin"], axis = 1) # Drop Cabin column

median_test_age = df_test_clean.Age.median() # Calc median Age from test set (naive)
df_test_clean.Age = df_test_clean.Age.fillna(median_test_age) # Replace test Ages with median

median_test_fare = df_test_clean.Fare.median() # Calc median Fare from test set (naive)
df_test_clean.Fare = df_test_clean.Age.fillna(median_test_fare)

df_test_clean.info()

# Convert str objects to floats in both train and test sets
le = preprocessing.LabelEncoder()
for column_name in df_train_clean.columns:
    if df_train_clean[column_name].dtype == object:
        df_train_clean[column_name] = le.fit_transform(df_train_clean[column_name])

for column_name in df_test_clean.columns:
    if df_test_clean[column_name].dtype == object:
        df_test_clean[column_name] = le.fit_transform(df_test_clean[column_name])

# Define our variables
y = df_train_clean["Survived"]
X = df_train_clean[["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]]

X_test = df_test_clean[["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]]

# Try scaling the data
X_scaled = X.copy(deep=True)
age_std_scaler = preprocessing.StandardScaler() # Standard scale the age
X_scaled[["Age"]] = age_std_scaler.fit_transform(X_scaled[["Age"]])

X_scaled.loc[X_scaled.Fare == 0, "Fare"] = 4.0125 # Replace zero fares with next minimum
X_scaled.Fare = 1/(X_scaled.Fare) # Transform fares to be reciprocal of fare
#X_scaled[["Fare"]] = np.log((X_scaled[["Fare"]])) # log transformation of fare, inactive

#sns.distplot((X.Fare))
#sns.distplot((X_scaled.Fare))

# Create and fit model
logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, y)

# Measure training accuracy
logistic_model.score(X_scaled, y)

# Predict on test set
y_hat = logistic_model.predict(X_test)

# Save csv in required Kaggle format
results = pd.DataFrame({"PassengerId": df_test_clean.PassengerId, "Survived": y_hat})
results.to_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/results2.csv", header = True, index = False)

