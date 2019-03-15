import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
sns.set()

##### data import #####
EastWest = pd.read_excel('C:/Users/nyh-0/Desktop/DM/EastWestAirlinesCluster_training.xls.xlsx')

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

##### PFA별 silhouette, davies_bouldin score graph plotting #####
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

for n in range(1, 11):
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
        davies_bouldin = davies_bouldin_score(x, cluster_labels)
        davies.append(davies_bouldin)
        sse = clusterer.inertia_

        print('- SSE =', sse)
        print('silhouette_score :', silhouette_avg)
        print('davies_bouldin :', davies_bouldin)
        print("\n")

    plt.plot(L, silhouette, 'mo-', label='silhouette')
    plt.plot(L, davies, 'bo-', label='davies')
    plt.xlabel('k')
    plt.ylabel('silhouette/davies')
    plt.title('silhouette/davies - when pfa is {}'.format(n), fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()
    print("============================")