# Método elbow para determinar a quantidade (k), de clusters ideais

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import matplotlib.pyplot as mat_Plt


class Elbow_Kmeans:

    def elbow(self, dataFrame):
        df = dataFrame

        X = np.array(df.drop(0,0))

        K = range(1, 10)

        distortions = []

        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

        mat_Plt.plot(K, distortions, 'bx-')
        mat_Plt.xlabel('k')
        mat_Plt.ylabel('Distorção')
        mat_Plt.title('Número de clusters ideais')
        mat_Plt.show()

df = pd.read_csv("train.csv")

Elbow_Kmeans = Elbow_Kmeans()
Elbow_Kmeans.elbow(dataFrame=df)

print("train.csv not found")
