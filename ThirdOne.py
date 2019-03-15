import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from math import sqrt
from operator import itemgetter
from scipy.spatial.distance import pdist, squareform

##### data import #####
EastWest = pd.read_excel('C:/Users/nyh-0/Desktop/DM/EastWestAirlinesCluster_pre2.xls.xlsx')

X = EastWest.drop(['ID#','cc2_miles','cc3_miles','Award?','Days_since_enroll'], axis=1)

##### 3개의 의미가 들어 잇지 않은 miles변수 구간 별 평균값 대체 #####
trans_miles_avg = [2500, 7500, 17500, 32500, 50000]
for i in range(1, 6):
    X = X.replace({'cc1_miles': i, 'cc2_miles': i, 'cc3_miles': i}, trans_miles_avg[i - 1])

Std = StandardScaler()
X = Std.fit_transform(X.astype(float))

silhouette = []
davies = []
K = range(2,11)
for k in K:
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette.append(silhouette_avg)

L = range(2,11)
for l in L:
    clusterer = KMeans(n_clusters=l, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    davies_bouldin = davies_bouldin_score(X, cluster_labels)
    davies.append(davies_bouldin)

plt.plot(K, silhouette, 'mo-', label = 'silhouette')
plt.plot(L, davies, 'bo-', label = 'davies')
plt.xlabel('k')
plt.ylabel('silhouette/davies')
plt.title('silhouette/davies - k', fontsize=14, fontweight='bold')
plt.legend()
plt.show()