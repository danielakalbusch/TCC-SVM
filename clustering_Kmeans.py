from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

class Clustering_Kmeans:

    def clustering(self, dataFrame):
        df = dataFrame

        X = np.array(df.drop('target', 1))

        kmeans = KMeans(n_clusters=5, random_state=0)

        kmeans.fit(X)

        df['K-classes'] = kmeans.labels_

        df.to_csv('train_clustered.csv')
        print("train_clustered.csv")

df = pd.read_csv("train.csv")
Clustering_Kmeans = Clustering_Kmeans()
Clustering_Kmeans.clustering(dataFrame=df)




