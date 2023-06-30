import pandas as pd
import math
from collections import Counter
import numpy as np


class KnnImplemented:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, p, q):
        distance = 0
        for i in range(len(p)):
            distance += (p[i] - q[i]) ** 2
        return math.sqrt(distance)

    def manhattan_distance(self, p, q):
        distance = 0
        for i in range(len(p)):
            distance += abs(p[i] - q[i])
        return distance

    def chebyshev_distance(self, p, q):
        diff = []
        for i in range(len(p)):
            diff.append(abs(p[i] - q[i]))
        return max(diff)

    def predict(self, x_test, k):
        y_predicted = []
        for i in range(len(x_test.index)):
            dists = []
            for j in range(len(self.x_train.index)):
                dist = self.chebyshev_distance(x_test.iloc[i], self.x_train.iloc[j])
                dists.append(dist)
            # sort descending all distances
            ed_df = pd.DataFrame(data=dists, columns=['distance'], index=self.y_train.index)
            ed_df_arr = ed_df.to_numpy()
            y_train_df = pd.DataFrame(data=self.y_train.copy(deep=True), index=self.y_train.index)
            y_train_df.insert(loc=0, column='distance', value=ed_df_arr)
            ed_df_knn = y_train_df.nsmallest(n=k, columns='distance', keep='first')
            y_prediction = ed_df_knn.loc[:, 'type'].mode()[0]
            y_predicted.append(y_prediction)
        return y_predicted
