import numpy as np
import pandas as pd

bank_df=pd.read_excel('C:\RPRO/bankdata.xlsx')

import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler

array = bank_df.values

# seperate array into input and output components
X = array[:,0:15]
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

# Compare Algorithms
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
# Prepare models
models = []
models.append((' LR ', LogisticRegression()))
models.append((' LDA', LinearDiscriminantAnalysis()))
models.append((' KNN ', KNeighborsClassifier()))
models.append((' CART ', DecisionTreeClassifier()))
models.append((' NB ', GaussianNB()))
models.append((' SVM ', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'f1'
hypothesis = xgb.sklearn.XGBClassifier(objective = "multi:softprob", max_depth=24, gamma=0.1, subsample = 0.90,
                               learning_rate=0.01, n_estimators=500, nthread=-1)
hypothesis.fit(X, Y, eval_set=[(X_valid, Y_valid)], eval_metric='merror',
                            verbose=False)
# for name, model in models:
#     kfold = KFold(n_splits=10, random_state=7)
#     cv_results = cross_val_score(hypothesis, X_trainval, Y_trainval, cv=kfold, scoring=scoring, n_jobs=1)
#     #cv_results = cross_val_score(model, X_trainval, Y_trainval, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

from sklearn.metrics import accuracy_score,confusion_matrix

print("accuracy : ", accuracy_score(X_test, hypothesis.pridict(Y_test)))
print(confusion_matrix(X_test, hypothesis.pridict(Y_test)))
# #ensemble
# hypothesis = AdaBoostClassifier(n_estimators=300, random_state=101)
# scores=cross_val_score(hypothesis, cv=3, scoring='f1', n_jobs=-1)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()