import pandas as pd
import pandas as pd

heartstatlog=pd.read_excel('C:/용호/한양대/3학년 2학기 2018/데이터마이닝/과제/과제5(~11.3)/Heartstatlog.xlsx')

import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
array = heartstatlog.values


X = heartstatlog.drop('Abs(1)/Pre(2)', axis=1)
Y = heartstatlog['Abs(1)/Pre(2)']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier


# fit model no training data
model = MLPClassifier()
model.fit(X_train, Y_train)

best_score = 0

for hidden_layer_sizes  in [(10,10,10),(30,30,30),(10,10),(100,100,100),(50,50)]:
    for alpha in [0.5,0.7,0.8,0.9,1.0,1.5]:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
    if score > best_score:
        best_score = score
        best_parameters = {'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha}


print('hidden_layer_sizes', hidden_layer_sizes, 'alpha', alpha)


model = MLPClassifier(**best_parameters)
model.fit(X_train,Y_train)
test_score = model.score(X_test, Y_test)
print("최고점수 : {:.2f}".format(best_score))
print("최적 매개변수", best_parameters)
print("최적 매개변수에서 테스트 점수 : {:.2f}".format(test_score))