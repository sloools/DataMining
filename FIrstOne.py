import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler

EastWest = pd.read_excel('C:/용호/한양대/3학년 2학기 2018\데이터마이닝\PBL/2차/EastWestAirlinesCluster_training.xls.xlsx')

X = EastWest.drop(['ID#', 'Award?'], axis=1)

##### 3개의 의미가 들어 잇지 않은 miles변수 구간 별 평균값 대체 #####
trans_miles_avg = [2500, 7500, 17500, 32500, 50000]
for i in range(1,6):
    X = X.replace({'cc1_miles':i, 'cc2_miles':i, 'cc3_miles':i}, trans_miles_avg[i-1])

scaler = StandardScaler()
X_Std = scaler.fit_transform(X.astype(float))

##### dendogram plotting #####
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.title("EastWest Dendograms")

dend = shc.dendrogram(shc.linkage(X_Std, method='ward'), truncate_mode='lastp')
plt.show()

##### KMeans의 최적의 k 값 결정(SSE elbow방법 사용) #####
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

SSE_list = []
K = range(2,11)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=10)
    kmeanModel.fit(X_Std)
    SSE_list.append(kmeanModel.inertia_)

##### SSE elbow에서 distance 구하기 위한 직선 식 생성 #####
slope = (SSE_list[-1] - SSE_list[0])/(K[-1] - K[0])
intercept = SSE_list[0] - K[0]*((SSE_list[-1] - SSE_list[0])/(K[-1] - K[0]))

a = SSE_list[-1] - SSE_list[0]
b = -K[-1] + K[0]
c = -K[0]*(SSE_list[-1] - SSE_list[0]) + SSE_list[0]*(K[-1] - K[0])

##### SSE elbow plotting #####
plt.plot(K, SSE_list, 'mo-')
plt.plot(K, slope*K + intercept, 'y')
plt.xlabel('k')
plt.ylabel('Sum of Squared Error')
plt.title('The Elbow Method', fontsize=14, fontweight='bold')
plt.show()

##### SSE elbow distance 계산하여 값이 큰 순으로 나열 #####
from math import sqrt
from operator import itemgetter

list =[]
for i in range(len(K)):
    d = abs(a*K[i] + b*SSE_list[i] + c)/sqrt(a**2 + b**2)
    list.append((i+2,d))
list.sort(key=itemgetter(1), reverse=True)
print('< 점(각 k값)과 직선 사이의 거리 >')
print(list)
print('k = 5일때, distance max')
print()

##### silhouette, davies_bouldin score graph plotting #####
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.metrics import davies_bouldin_score

silhouette = []
davies = []
K = range(2,11)
for k in K:
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(X_Std)

    silhouette_avg = silhouette_score(X_Std, cluster_labels)
    silhouette.append(silhouette_avg)

L = range(2,11)
for l in L:
    clusterer = KMeans(n_clusters=l, random_state=10)
    cluster_labels = clusterer.fit_predict(X_Std)

    #davies_bouldin = davies_bouldin_score(X_Std, cluster_labels)
    #davies.append(davies_bouldin)

plt.plot(K, silhouette, 'mo-', label = 'silhouette')
plt.plot(L, davies, 'bo-', label = 'davies')
plt.xlabel('k')
plt.ylabel('silhouette/davies')
plt.title('silhouette/davies - k', fontsize=14, fontweight='bold')
plt.legend()
plt.show()

print('< 군집화 성능 >')
print('k = 5일때, silhouette, davies 성능 모두 좋음')
print()
print('silhouette_score :', silhouette[3])
print('davies_bouldin_score :', davies[3])
print('SSE :', SSE_list[3])