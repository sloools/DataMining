import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
sns.set()
from sklearn.preprocessing import StandardScaler

EastWest = pd.read_excel('C:/용호/한양대/3학년 2학기 2018\데이터마이닝\PBL/2차/EastWestAirlinesCluster_training.xls.xlsx')

X = EastWest.drop(['ID#'], axis=1)

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


from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.metrics import davies_bouldin_score

for n in range(1, 6):
    pfa = PFA(n_features=n)
    pfa.fit(X)
    x = pfa.features_
    column_indices = pfa.indices_
    print("PFA 값이 {} 일때 추출된 컬럼 : {}".format(n,column_indices))

    silhouette = []
    davies = []
    K = range(2,11)
    print("=========PFA : {} ==========".format(n))
    for k in K:
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(x)

        silhouette_avg = silhouette_score(x, cluster_labels)

        sse = clusterer.inertia_
        print("군집화 k 의 개수 : {}".format(k))
        print('- SSE =', sse)
        print('silhouette_score :', silhouette_avg)
        print("\n")
    print("============================")