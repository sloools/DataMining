import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from collections import Counter
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

EastWest = pd.read_excel('C:/용호/한양대/3학년 2학기 2018\데이터마이닝\PBL/2차/EastWestAirlinesCluster_training.xls.xlsx')
X = EastWest.drop(['ID#', 'Award?'], axis=1)

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, Y):
        if not self.q:
            self.q = Y.shape[1]

        Std = StandardScaler()
        Y = Std.fit_transform(Y)

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

pfa = PFA(n_features=4)
pfa.fit(X)
x = pfa.features_
column_indices = pfa.indices_

from sklearn.metrics import silhouette_samples, silhouette_score

# from sklearn.metrics import davies_bouldin_score

silhouette = []
davies = []
K = [2,6]
for k in K:
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(x)

    dist = clusterer.cluster_centers_

    silhouette_avg = silhouette_score(x, cluster_labels)

    sse = clusterer.inertia_
    print('- SSE =', sse)
    print('silhouette_score :', silhouette_avg)
    print("\n")

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