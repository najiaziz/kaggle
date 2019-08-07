"""

Titanic: Machine Learning from Disaster
Date started: 7th August 2019
Author: Naji Aziz

"""

# Import basic modules
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Pull data
df_train = pd.read_csv("C:\\Users\\infra\\Documents\\KAGGLE\\Titanic\\train.csv")
df_test = pd.read_csv("C:\\Users\\infra\\Documents\\KAGGLE\\Titanic\\test.csv")

# Brief exploration of data
df_train.info()
df_train.head()
df_train.tail()
df_test.info()