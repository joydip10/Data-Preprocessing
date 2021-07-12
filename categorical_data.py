# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
#onehotencoder = OneHotEncoder(categories = 'auto')
#X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable

X = pd.concat([pd.get_dummies(X['Country']),X],axis=1)
X=X.drop(['Country'],axis=1)
X=X.values

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)