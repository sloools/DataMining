import numpy as np
import pandas as pd
bank_df=pd.read_excel('C:/RPRO/DM/sample_less_column.xlsx')


import numpy
from sklearn.preprocessing import MinMaxScaler
# from sklearn.cross_validation import import cross_val_score, StratifiedkFold
from sklearn.ensemble import GradientBoostingClassifier

array = bank_df.values

# seperate array into input and output components
X = array[:,0:12]
Y = bank_df['y']
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
numpy.set_printoptions(precision=3)
#print(rescaledX[0:])

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Compare Algorithms


hypothesis = GradientBoostingClassifier(max_depth=5,learning_rate= 0.5 ,
                               random_state=101, n_estimators=50)
hypothesis.fit(X,Y)


print("훈련세트정확도 : {:.3f}".format(hypothesis.score(X_train, Y_train)))
print("테스트세트정확도 : {:.2f}".format(hypothesis.score(X_test, Y_test)))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

cv_results = cross_val_score(hypothesis, X, Y, cv=3, scoring='f1')
msg = " %f (%f)" % ( cv_results.mean(), cv_results.std())
print(msg)
# from sklearn.metrics import accuracy_score,confusion_matrix
#
# print(confusion_matrix(X_test, forest.score(X_test, Y_test)))
