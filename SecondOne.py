import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.cm as cm
import numpy as np
sns.set()

##### data import #####
EastWest = pd.read_excel('C:/용호/한양대/3학년 2학기 2018\데이터마이닝\PBL/2차/EastWestAirlinesCluster_training.xls.xlsx')

X = EastWest.drop(['ID#'], axis=1)

##### 3개의 의미가 들어 잇지 않은 miles변수 구간 별 평균값 대체 #####
trans_miles_avg = [2500, 7500, 17500, 32500, 50000]
for i in range(1,6):
    X = X.replace({'cc1_miles':i, 'cc2_miles':i, 'cc3_miles':i}, trans_miles_avg[i-1])

##### PFA 방법을 이용하여 Feature 추출 #####
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, Y):
        if not self.q:
            self.q = Y.shape[1]

        Std = StandardScaler()
        Y = Std.fit_transform(Y.astype(float))

        A = Y.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A)
        clusters = kmeans.predict(A)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = Y[:, self.indices_]

##### PFA별 silhouette, davies_bouldin score graph plotting #####
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.metrics import davies_bouldin_score
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

for n in range(1, 5):
    pfa = PFA(n_features=n)
    pfa.fit(X)
    x = pfa.features_
    column_indices = pfa.indices_
    print("PFA 값이 {} 일때 추출된 컬럼 : {}".format(n,column_indices))

    silhouette = []
    davies = []
    L = range(2, 11)
    print("=========PFA : {} ==========".format(n))
    for l in L:
        clusterer = KMeans(n_clusters=l, random_state=10)
        cluster_labels = clusterer.fit_predict(x)

        silhouette_avg = silhouette_score(x, cluster_labels)
        silhouette.append(silhouette_avg)
        #davies_bouldin = davies_bouldin_score(x, cluster_labels)
        #davies.append(davies_bouldin)
        sse = clusterer.inertia_

        print('SSE : ', sse)
        print('silhouette_score :', silhouette_avg)
        #print('davies_bouldin :', davies_bouldin)
        print("\n")

    plt.plot(L, silhouette, 'mo-', label='silhouette')
    plt.plot(L, davies, 'bo-', label='davies')
    plt.xlabel('k')
    plt.ylabel('silhouette/davies')
    plt.title('silhouette/davies - when pfa is {}'.format(n), fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()
    print("============================")


# 선택된 pfa 와 k

pfa = PFA(n_features=4)
pfa.fit(X)
x = pfa.features_
column_indices = pfa.indices_

clusterer = KMeans(n_clusters=6, random_state=10)
cluster_labels = clusterer.fit_predict(x)

dist = clusterer.cluster_centers_

silhouette_avg = silhouette_score(x, cluster_labels)
silhouette.append(silhouette_avg)
#davies_bouldin = davies_bouldin_score(x, cluster_labels)
#davies.append(davies_bouldin)
sse = clusterer.inertia_

print('PFA 4, k = 6 일때 SSE : ', sse)
print('PFA 4, k = 6 일때 silhouette_score :', silhouette_avg)
#print('PFA 4, k = 6 일때 davies_bouldin :', davies_bouldin)
print("\n")


##### KMeans의 최적의 k 값 결정(SSE elbow방법 사용) #####
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

SSE_list = []
K = range(2,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=10)
    kmeanModel.fit(X)
    SSE_list.append(kmeanModel.inertia_)

#SSE elbow에서 distance 구하기 위한 직선 식 생성
slope = (SSE_list[-1] - SSE_list[0])/(K[-1] - K[0])
intercept = SSE_list[0] - K[0]*((SSE_list[-1] - SSE_list[0])/(K[-1] - K[0]))

a = SSE_list[-1] - SSE_list[0]
b = -K[-1] + K[0]
c = -K[0]*(SSE_list[-1] - SSE_list[0]) + SSE_list[0]*(K[-1] - K[0])

# SSE elbow plotting
plt.plot(K, SSE_list, 'mo-')
plt.plot(K, slope*K + intercept, 'y')
plt.xlabel('k')
plt.ylabel('Sum of Squared Error')
plt.title('The Elbow Method', fontsize=14, fontweight='bold')
plt.show()

# SSE elbow distance 계산하여 값이 큰 순으로 나열
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



###### 댄드로그램 ######
import scipy.cluster.hierarchy as shcu
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.title("EastWest Dendograms")

dend = shc.dendrogram(shc.linkage(X, method='ward'), truncate_mode='lastp')
plt.show()

####### 군집화 플랏 #####
range_n_clusters = [6]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x)

    #davies_bouldin = davies_bouldin_score(x, cluster_labels)
    #print("For n_clusters =", n_clusters,
          #"The davies_bouldin_score is :", davies_bouldin)

    silhouette_avg = silhouette_score(x, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(x, cluster_labels)

    SSE = clusterer.inertia_
    print("For n_clusters =", n_clusters,
          "The SSE is :", SSE)
    print()

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        ax1.text(0.7, y_lower + 0.5 * size_cluster_i, str(np.round(ith_cluster_silhouette_values.mean(), decimals=2)))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="r", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(x[:, 0], x[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_

    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
plt.show()


####### 정량평가 ######
# 군집간 거리 구하기
from scipy.spatial.distance import pdist, squareform

print("< 정량 평가 1 - 군집 간 거리 >\n")
distance = pdist(dist, 'euclidean')
df_dist = pd.DataFrame(squareform(distance))
print(df_dist)

# 군집의 크기(개수) 구하기
c = Counter(clusterer.labels_)
numOfinstance = sorted(c.items(), key=lambda pair: pair[0], reverse=False)

# 최대 거리 계산

dists = euclidean_distances(dist, x)

centerTOrecord = []
for i in range(len(dist)):
    maxdist = max(dists[i, clusterer.labels_ == i])
    centerTOrecord.append(maxdist)

# 데이터 프레임으로 병합

df_size = pd.DataFrame(numOfinstance)
df_size = df_size.drop(0, axis=1)
df_size[2] = centerTOrecord
df_size.columns = ["군집의 크기(개수)", "Center-Record 최대거리"]
print("\n< 정량 평가 2 >\n")
print(df_size)
