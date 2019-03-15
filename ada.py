import numpy as np
import pandas as pd


import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler

bank_df = pd.read_excel('C:/RPRO/DM/cho/bank_data_training_preprocessed_oversample.xlsx')
test_df = pd.read_excel('C:/RPRO/DM/cho/bank_data_finaltest_preprocessed.xlsx')


X = bank_df.drop('y', axis=1)
Y = bank_df['y']
X_test = test_df.drop('y', axis=1)
Y_true = test_df['y']

# summarize transformed data
numpy.set_printoptions(precision=3)
#print(rescaledX[0:])



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier




estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

estimator.fit(X,Y)


print("훈련세트정확도 : {:.3f}".format(estimator.score(X, Y)))


from sklearn.model_selection import cross_val_score


cv_results = cross_val_score(estimator, X, Y, cv=3, scoring='f1')
msg = " %f (%f)" % ( cv_results.mean(), cv_results.std())
print(msg)

y_pred = estimator.predict(X_test)
# Y의 예측값을 변수로 지정하여 모델의 정확도 측정
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(Y_true, y_pred))
print(confusion_matrix(Y_true, y_pred))

'''
bank_df=pd.read_excel('C:/RPRO/DM/sample_less_column.xlsx')

import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler

array = bank_df.values

# seperate array into input and output components
X = array[:,0:14]
Y = bank_df['y']
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
numpy.set_printoptions(precision=3)
#print(rescaledX[0:])

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_trainval,Y_trainval,random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier




estimator = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, learning_rate=1.0,algorithm='SAMME', random_state=None)

estimator.fit(X_trainval,Y_trainval)


print("훈련세트정확도 : {:.3f}".format(estimator.score(X_train, Y_train)))
print("테스트세트정확도 : {:.2f}".format(estimator.score(X_test, Y_test)))


from sklearn.model_selection import cross_val_score


cv_results = cross_val_score(estimator, X, Y, cv=3, scoring='f1')
msg = " %f (%f)" % ( cv_results.mean(), cv_results.std())
print(msg)
'''
