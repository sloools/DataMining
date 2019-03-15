# 라이브러리, 데이터 불러오기
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

customer_data = pd.read_csv('C:\RPRO/shopping_data.csv')


customer_data.shape                                         # 현재 데이터 모양 보기

print(customer_data.head())                                 # 데이터 head 출력

data = customer_data.iloc[:, 3:5].values                    # 1~3열을 제외하고, Annual income과 spending score만 고려

import scipy.cluster.hierarchy as shc                       # HC 모델 불러오기

plt.figure(figsize=(10, 7))                                 # 가로 10, 세로 7로 그림 그리기
plt.title("Customer Dendograms")                            # title 은 "customer Dendrogram"
dend = shc.dendrogram(shc.linkage(data, method='ward'))     # 그룹을 만드는 선을 잇는 방법으로 "ward"(ward는 분산을 최소화하는 알고리즘 방법)
plt.show()

