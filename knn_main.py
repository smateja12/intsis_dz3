import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sb

from knn import KnnImplemented

import math


def type_param_dependency(column):
    sb.set(rc={"figure.figsize": (8, 4)})
    np.random.seed(0)
    cont_params_df = data.loc[data[column].notnull()]
    sb.displot(data=cont_params_df, x=column, hue='type', multiple='stack').set(title='type dependency of ' + column)
    # sb.displot(data=cont_params_df, x='butter', hue='type', kind='kde')
    plt.show()


if __name__ == "__main__":

    # 1. PROBLEM STATEMENT AND READ DATA

    # show all columns
    pd.set_option('display.max_columns', 8)
    # show content on full screen width
    pd.set_option('display.width', None)

    data = pd.read_csv('datasets/cakes.csv')

    # 2. DATA ANALYSIS

    print("Prvih 5 redova:")
    print(data.head())
    print("===================================")

    print("Data profiling:")
    print(data.info())
    print("===================================")

    print("Feature statistic:")
    print(data.describe())
    print(data.describe(include=[object]))
    print("===================================")

    # 3. DATA CLEANSING AND DATA REPRESENTATION
    type_param_dependency('butter')

    # 4. FEATURE ENGINEERING
    # not useful features:
    # useful features: flour,eggs,sugar,milk,butter,baking_powder

    # Encode type feature to 0/1 values
    le = LabelEncoder()
    type_new_values = le.fit_transform(data['type'])
    data['type'] = type_new_values

    data_train = data.loc[:, ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']] # DataFrame

    labels = data.loc[:, 'type']  # Series
    # print(data_train.head())

    # Scale all columns values from 0 to 1
    # new_val = (old_val - min_val) / (max_val - min_val)
    d1 = (data_train - data_train.min())
    d2 = (data_train.max() - data_train.min())
    data_train_scaled = d1 / d2
    print(data_train_scaled.head())

    # Split data_train_scaled
    X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, labels, test_size=0.3, random_state=40, shuffle=True)

    num_of_neighbors = math.ceil(np.sqrt(len(data.index)))

    knn_model = KNeighborsClassifier(n_neighbors=num_of_neighbors)
    knn_model.fit(X_train, y_train)
    labels_predicted = knn_model.predict(X_test)
    y_pred = pd.Series(data=labels_predicted, name='type_Predicted', index=X_test.index)
    result_dataframe = pd.concat([X_test, y_test, y_pred], axis=1)

    # Python KNN SCORE
    # print("Python KNN RESULT DF:")
    # print(result_dataframe.head())
    print(f'Python KNN score: {knn_model.score(X_test, y_test):.6f}')


    # KnnImplemented model
    data_train = data.loc[:, ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]  # DataFrame

    labels = data.loc[:, 'type']  # Series

    d1 = (data_train - data_train.min())
    d2 = (data_train.max() - data_train.min())
    data_train_scaled = d1 / d2

    X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, labels, test_size=0.25, random_state=40, shuffle=True)

    knn_implemented = KnnImplemented(X_train, y_train)
    y_predicted = knn_implemented.predict(x_test=X_test, k=num_of_neighbors)
    print(f'KnnImplemented ACC: {accuracy_score(y_test, y_predicted):.6f}')
    # print(f'KnnImplemented R2 score: {r2_score(y_test, y_predicted):.6f}')

    # correlation matrix
    sb.heatmap(data_train.corr(), annot=True, square=True, fmt='.2f')
    plt.show()
