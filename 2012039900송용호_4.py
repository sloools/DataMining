#과제 3-4

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn import svm

data = load_iris()

feature = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'o'
    elif t == 2:
        c = 'b'
        marker = 'x'

    plt.scatter(feature[target ==t,3],
                feature[target ==t,2],
                marker=marker,
                c=c)
    plt.ylabel('petal_length')
    plt.xlabel('petal_width')

pl_length =  feature[:,2]
labels = target_names[target]
is_setosa = (labels == 'setosa')
max_setosa = pl_length[is_setosa].max()
min_non_setosa = pl_length[~is_setosa].min()
print('Maximum of setosa : {0}\n'
      'Minimum of others : {1}\n'.format(max_setosa,min_non_setosa))
mid_line=(max_setosa+min_non_setosa)/2


x = np.arange(0,2.5) #첫번째 라인 그리기
y = -0.7*x + mid_line
plt.plot(x,y,'b')

features = feature[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')

best_acc = 0.0
for feature in range(features.shape[1]):
    threshold = features[:, feature]
    for t in threshold:
        feature_i = features[:, feature]
        pred = (feature_i > t)
        acc = (pred == is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc > acc:
            reverse = True
            acc = rev_acc
        else:
            reverse = False
        if acc > best_acc:
            best_acc = acc
            best_fi = feature
            best_t = t
            best_reverse = reverse

if best_fi == 0:
    best_fi = 'Sepal_length'
elif best_fi ==1:
    best_fi = 'Sepal_width'
elif best_fi ==2:
    best_fi = 'Petal_length'
elif best_fi ==3:
    best_fi = 'Petal_width'
print('virginica의 분류 기준 : ',best_fi,'\n',
                                    best_fi,'의 경계값 : ',best_t)


plt.axvline(x=best_t,ymin=0.25,ymax=0.9, color ='b') #두번째 라인 그리기


plt.show()


