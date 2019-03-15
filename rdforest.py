import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

# 엑셀 데이터셋 불러오기 및 변수 그룹화
bank_df = pd.read_excel('C:/RPRO/DM/cho/bank_data_training_preprocessed_oversample.xlsx')
test_df = pd.read_excel('C:/RPRO/DM/cho/bank_data_finaltest_preprocessed.xlsx')

X = bank_df.drop('y', axis=1)
Y = bank_df['y']
X_test = test_df.drop('y', axis=1)
Y_true = test_df['y']

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models=[]
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2, max_depth=15,  min_samples_leaf=6)
forest.fit(X, Y)

y_pred = forest.predict(X_test)
# Y의 예측값을 변수로 지정하여 모델의 정확도 측정

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(Y_true, y_pred))
print(confusion_matrix(Y_true, y_pred))
'''
bank_df=pd.read_excel('C:/RPRO/DM/sample_less_column.xlsx')
test=pd.read_excel('C:\RPRO/bankdata3.xlsx')

import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler

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

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=5, random_state=2, max_depth=15,  min_samples_leaf=6)
forest.fit(X_train, Y_train)
print("훈련세트정확도 : {:.3f}".format(forest.score(X_train, Y_train)))
print("테스트세트정확도 : {:.2f}".format(forest.score(X_test, Y_test)))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

cv_results = cross_val_score(forest, X, Y, cv=3, scoring='f1')
msg = " %f (%f)" % ( cv_results.mean(), cv_results.std())
print(msg)
# from sklearn.metrics import accuracy_score,confusion_matrix
#
# print(confusion_matrix(X_test, forest.score(X_test, Y_test)))
'''