import pandas as pd

heartstatlog = pd.read_excel('C:/용호/한양대/3학년 2학기 2018/데이터마이닝/과제/과제5(~11.3)/Heartstatlog.xlsx')

from sklearn.model_selection import train_test_split



X = heartstatlog.drop('Abs(1)/Pre(2)', axis=1)
Y = heartstatlog['Abs(1)/Pre(2)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

sc = StandardScaler()
mlc = MLPClassifier(random_state=1,nesterovs_momentum=True)
loo = LeaveOneOut()
pipe = make_pipeline(sc, mlc)

parameters = {"mlpclassifier__hidden_layer_sizes":[(168,),(126,),(498,),(166,)],"mlpclassifier__solver" : ('sgd','adam'), "mlpclassifier__alpha": [0.001,0.0001],"mlpclassifier__learning_rate_init":[0.005,0.001] }
clf = GridSearchCV(pipe, parameters,n_jobs= -1,cv = loo)
clf.fit(X, Y)
model = clf.best_estimator_
print("the best model and parameters are the following: {} ".format(model))