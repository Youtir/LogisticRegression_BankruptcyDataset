#librairies import

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score
#importing the dataset as a pandas dataframe
raw_data = pd.read_csv('D:\Qualitative_Bankruptcy\Qualitative_Bankruptcy.data.csv')
raw_data.head()
#Labels encoding

labelencoder = LabelEncoder()
for col in raw_data.columns:
    raw_data[col] = labelencoder.fit_transform(raw_data[col])

#Separation of variables and target data 
x = raw_data.iloc[:,0:6]
y = raw_data.iloc[:,6]

x.head()
y.head()
#Splitting data rows into test data and training data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#Training the Logistic Regression Model

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_prob = lr.predict(x_test)

#Obtaining the model accuracy

accuracy_score(y_test, y_prob)
# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE

# creating the RFE model and select the main attribute
rfe = RFE(lr, 1)
rfe = rfe.fit(x_train, y_train)
# summarizing the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
'''it appears that the categories or risk of bankruptcy are sorted by importance as following :
Competitiveness, Operating-Risk, Credibility, Financial-Flexibility, Management-Risk, Industrial-Risk'''