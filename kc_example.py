from __future__ import print_function
from sklearn.datasets import make_blobs                             #sklearn을 이용하여 가상 데이터 생성
from sklearn.cluster import KMeans                                  #Kmeans clustering 라이브러리
from sklearn.metrics import silhouette_samples, silhouette_score    #실루엣 계수 계산 라이브러리

import matplotlib.pyplot as plt
import matplotlib.cm as cm                                          # plot을 위한 라이브러리
import numpy as np

print(__doc__)
                                                                    # "pbl 문제 때는 여기에 제공한 엑셀 파일을 넣어야합니다."
X, y = make_blobs(n_samples=500,                                    # 샘플 갯수
                  n_features=4,                                     # feature 수
                  centers=4,                                        # 중심 갯수
                  cluster_std=1,                                    # 군집간 편차
                  center_box=(-10.0, 10.0),                         # 그림 크기
                  shuffle=True,                                     # 섞기
                  random_state=1)

range_n_clusters = [2, 3, 4, 5, 6]                                  # 만들 군집 수의 범위 2개~6개

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)                            #1X2 로 그림 출력
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)      #10개의 seed(random_state)로 10번 재생산
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)            #실루엣 점수 평균(클래스 label당)
    print("For n_clusters =", n_clusters,                           # 군집이 n개 일때,
          "The average silhouette_score is :", silhouette_avg)      # 실루엣 점수가 몇점이다.

    sample_silhouette_values = silhouette_samples(X, cluster_labels)
                                                                    #41~56 실루엣 계수가 들어갈 그림을 만드는 과정입니다.
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

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))         # 군집에 붙는 수 정하기

        y_lower = y_upper + 10  # 10 for the 0 samples                  # 0부터 시작해서 10씩 더해가면서

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")                 # x축은 실루엣 계수 값
    ax1.set_ylabel("Cluster label")                                     # y축은 군집 label

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])                    # 간격

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,    # s : lw : line의 폭, alpha : 0~1로 1에 가까울수록 불투명
                c=colors, edgecolor='k')

                                                                        # 클러스터에 라벨링하기
    centers = clusterer.cluster_centers_

    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',               # 중심에 하얀 원으로 표시
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