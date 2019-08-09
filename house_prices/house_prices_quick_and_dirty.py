"""

House Prices: Advanced Regression Techniques
Date started: 9th August 2019
Author: Naji Aziz

"""

import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

raw_train = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/house_prices/train.csv")
raw_test = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/house_prices/test.csv")

raw_train["IsTrain"] = 1
raw_test["IsTrain"] = 0
raw_test["SalePrice"] = 0

raw_all = pd.concat([raw_train, raw_test], axis = 0, sort=False)
'''
raw_all.info()
raw_all.tail()

raw_train.info()
raw_test.info()
'''

df_all = raw_all.dropna(axis = 1) # drop all columns with missing values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column_name in df_all.columns:
    if df_all[column_name].dtype == object:
        df_all[column_name] = le.fit_transform(df_all[column_name])

df_train = df_all[df_all.IsTrain == True].drop("IsTrain", axis = 1)
df_test = df_all[df_all.IsTrain == False].drop("IsTrain", axis = 1)

X = df_train.drop("SalePrice", axis = 1)
y = df_train.SalePrice

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)

rf.score(X,y)


# now predict on test
X_test = df_test.drop("SalePrice", axis = 1)
y_hat = rf.predict(X_test)

results = pd.DataFrame({"Id": X_test.Id, "SalePrice": y_hat})
results.to_csv("C:/Users/infra/Documents/kaggle/kaggle-master/house_prices/results1.csv", header = True, index = False)
