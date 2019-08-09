"""

Titanic: Machine Learning from Disaster
Date started: 7th August 2019
Author: Naji Aziz

"""

# Import basic modules
import pandas as pd
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

# Set some options
pd.set_option('display.expand_frame_repr', False)
sns.set(color_codes=True, rc={'figure.figsize':(20,15)})

# Pull data into environment
df = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/train.csv")
df_test = pd.read_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/test.csv")

# Add some columns for joining and later detaching
df["IsTrain"] = 1
df_test["IsTrain"] = 0
df_test["Survived"] = 0 # to be changed later

# Combine datasets
df_comb = pd.concat([df, df_test], axis = 0, sort=False)

# Filling in Age
df_comb["ChildTitle"] = (df_comb.Name.str.extract(' ([A-Za-z]+)\.') == "Master") | (df_comb.Name.str.extract(' ([A-Za-z]+)\.') == "Miss")
ageFirstQuartile, ageThirdQuartile = df_comb.Age.quantile([0.1,0.9])
df_comb.loc[(df_comb.Age.isna()==True) & (df_comb.ChildTitle==True), "Age"] = ageFirstQuartile
df_comb.loc[(df_comb.Age.isna()==True) & (df_comb.ChildTitle==False), "Age"] = ageThirdQuartile

# Filling in Embarked and Fare
#df_comb.loc[(df_comb.Fare.isna()|df_comb.Embarked.isna()),:]
#df_comb.Embarked.value_counts() #Establish that S is most common embark category, so replace with S
#df_comb.Fare.median() #Replace missing Fare with median fare
df_comb.loc[df_comb.Fare.isna(),:"Fare"] = df_comb.Fare.median()
df_comb.loc[df_comb.Embarked.isna(), "Embarked"] = "S"

# For now, drop cabin as too sparse
df_comb = df_comb.drop("Cabin", axis = 1)
# Also drop other redundant columns
df_comb = df_comb.drop(["Name", "ChildTitle", "Ticket"], axis = 1)

# Convert objects to floats
le = preprocessing.LabelEncoder()
for column_name in df_comb.columns:
    if df_comb[column_name].dtype == object:
        df_comb[column_name] = le.fit_transform(df_comb[column_name].astype(str))

#View distributions
#sns.distplot(df_comb["Fare"].dropna())

# Scale and normalize features
# Age is normally distributed, but Fare SibSp and Parch are positively skewed
ageStdScaler = preprocessing.StandardScaler()
df_comb[["Age"]] = ageStdScaler.fit_transform(df_comb[["Age"]])

# Transforming Fare
df_comb.Fare += 3.1708 # the minimum fare someone paid, above 0: df_comb[df_comb.Fare!=0].Fare.min()
df_comb.Fare = np.log(df_comb.Fare) # log transforming
#df_comb.Fare = 1/df_comb.Fare # could also do reciprocal, but choosing not to

# Split back into train and test
df = df_comb.loc[df_comb.IsTrain == True, :].drop(["IsTrain"], axis=1)

df_test = df_comb.loc[df_comb.IsTrain == False, :].drop(["Survived","IsTrain"], axis=1)
X_test = df_test.drop(["PassengerId"], axis =1)

X = df.drop(["PassengerId","Survived"], axis = 1)
y = df["Survived"]

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.3, random_state = 1912)

# trying logistic regression

lm = LogisticRegression()
lm.fit(X_train, y_train)

lm.score(X_train, y_train)
lm.score(X_dev, y_dev)

y_hat_logit = lm.predict(X_test)

#trying random forest
rf = RandomForestClassifier(max_features = 4, min_samples_leaf = 1, min_samples_split = 4, n_estimators = 50)
rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_dev, y_dev)

# now fit rf on full training data set, before making final predictions
rf.fit(X, y)

y_hat_rf = rf.predict(X_test)

''' RF hyperparameter tweaking
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5], "min_samples_split" : [3, 4, 5, 6], "max_features" : [3, 4, 5], "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs.fit(X_train, y_train)
'''


'''Trying SVM
'''

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_dev, y_dev)
svc.fit(X, y)

y_hat_svc = svc.predict(X_test)


df_test.PassengerId = df_test.PassengerId.astype(int)
df_test.loc[df_test.PassengerId == 14,"PassengerId"] = 1044 # odd correction needed
df_test.info()
# Save csv in required Kaggle format
results = pd.DataFrame({"PassengerId": df_test.PassengerId, "Survived": y_hat_svc.astype(np.int)})
results.to_csv("C:/Users/infra/Documents/kaggle/kaggle-master/titanic/results3.csv", header = True, index = False)
