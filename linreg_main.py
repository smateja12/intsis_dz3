# Uvoz pandas modula za manipulaciju nad podacima.
# Alias pd za pandas se koristi po konvenciji.
import pandas as pd

# Uvoz pyplot modula za vizuelizaciju podataka.
# Alias plt za pyplot se koristi po konvenciji.
import matplotlib.pyplot as plt

# Uvoz numpy modula za rad sa visedimenzionim nizovima.
# Alias np za numpy se koristi po konvenciji.
import numpy as np

# Mapa boja (colormap) za bojenje funkcije greske
from matplotlib import cm

# Obican model Linearne regresije
from sklearn.linear_model import LinearRegression

from LinearRegression import LinearRegressionGradientDescent

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale
from sklearn.metrics import r2_score

import seaborn as sb


def co2_cont_param_dependency(column):
    # continuos data: MODELYEAR, ENGINESIZE, CYLINDERS, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB, FUELCONSUMPTION_COMB_MPG

    plt.figure('CO2Emissions CONT Param')
    X = data[[column, 'CO2EMISSIONS']]
    # X = X.sort_values(by=column)

    y = data['CO2EMISSIONS']
    y = y / 10

    plt.scatter(X[column], y, s=23, c='red', marker='o', alpha=0.7,
                edgecolors='black', linewidths=2, label='CO2EMISSIONS dependency')
    plt.legend()
    plt.xlabel(column)
    plt.ylabel('CO2EMISSIONS')
    plt.title('CO2EMISSIONS dependence of ' + column)
    plt.show()


def co2_cat_param_dependecy(column):
    # categorial data: MAKE, MODEL, VEHICLECLASS, TRANSMISSION, FUELTYPE

    plt.figure('CO2Emissions CAT Param')
    plt.xlabel(column)
    plt.ylabel('CO2EMISSIONS MEDIAN VALUES')
    plt.title('CO2EMISSIONS dependence of ' + column)
    cat_param_grouped = data.groupby(column)
    co2emissions_values = cat_param_grouped['CO2EMISSIONS']
    co2emissions_mean_values = co2emissions_values.median()
    co2emissions_mean_values.plot.bar()
    plt.show()


if __name__ == "__main__":

    # 1. PROBLEM STATEMENT AND READ DATA

    # show all columns
    pd.set_option('display.max_columns', 13)
    # show content on full screen width
    pd.set_option('display.width', None)

    # Citanje .csv fajla i kreiranje DataFrame-a od njega.
    # Zaglavlje .csv fajla predstavlja imena kolona DataFrame-a.
    data = pd.read_csv('datasets/fuel_consumption.csv')

    # 2. DATA ANALYSIS

    print("Prvih 5 redova:")
    print(data.head(5))
    print("===================================")

    print("Data profiling:")
    print(data.info())
    print("===================================")

    print("Feature statistic:")
    print(data.describe())
    print(data.describe(include=[object]))
    print("===================================")

    # 3. DATA CLEANSING AND DATA REPRESENTATION

    # fill NaNs values with the mean value in the ENGINESIZE column(float 64)
    data['ENGINESIZE'] = data['ENGINESIZE'].fillna(data['ENGINESIZE'].mean())
    # print(data.loc[data['ENGINESIZE'].isnull()].head())

    # fill NaNs values with the most frequent(mode) value in the TRANSMISSION column(object)
    data['TRANSMISSION'] = data['TRANSMISSION'].fillna(data['TRANSMISSION'].mode()[0])
    # print(data.loc[data['TRANSMISSION'].isnull()].head())

    # fill NaNs values with the most frequent(mode) value in the TRANSMISSION column(object)
    data['FUELTYPE'] = data['FUELTYPE'].fillna(data['FUELTYPE'].mode()[0])
    # print(data.loc[data['FUELTYPE'].isnull()].head())

    # delete NaNs rows
    # data.where(data['FUELTYPE'].notnull(), inplace=True)


    # Graphical representation of the dependence of CO2EMISSIONS on each CONTINUOUS data
    co2_cont_param_dependency('ENGINESIZE')

    # # Graphical representation of the CO2EMISSIONS of each continual data
    # co2_cont_param_dependency('CYLINDERS')

    # # Graphical representation of the CO2EMISSIONS dependence of FUELCONSUMPTION_CITY
    # co2_cont_param_dependency('FUELCONSUMPTION_CITY')

    # # Graphical representation of the CO2EMISSIONS dependence of FUELCONSUMPTION_HWY
    # co2_cont_param_dependency('FUELCONSUMPTION_HWY')

    # # # Graphical representation of the CO2EMISSIONS dependence of FUELCONSUMPTION_COMB
    # co2_cont_param_dependency('FUELCONSUMPTION_COMB')

    # # # Graphical representation of the CO2EMISSIONS dependence of FUELCONSUMPTION_COMB_MPG
    # co2_cont_param_dependency('FUELCONSUMPTION_COMB_MPG')

    # Graphical representation of the dependence of CO2EMISSIONS on each  data
    co2_cat_param_dependecy('FUELTYPE')


    # 4. FEATURE ENGINEERING
    # not useful features: MODELYEAR, MAKE, MODEL, VEHICLECLASS, TRANSMISSION, FUELCONSUMPTION_COMB_MPG
    # useful features: ENGINESIZE, CYLINDERS, FUELTYPE, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB, CO2EMISSIONS

    data_train = data[['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB_MPG']] # DataFrame
    labels = data.loc[:, 'CO2EMISSIONS'] #Series
    # print(data_train.head())

    # kodiranje 2 vrednosti
    # le = LabelEncoder()
    # transmission_new_values = le.fit_transform(data_train['TRANSMISSION'])
    # for i in range(len(transmission_new_values)):
    #     data_train.loc['TRANSMISSION'] = transmission_new_values[i]
    # print(data_train.head())

    ohe = OneHotEncoder(dtype=int, sparse_output=False)
    fueltype = ohe.fit_transform(data_train['FUELTYPE'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['FUELTYPE'])
    data_train = data_train.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE'])))
    # print(data_train.head())

    # Scale all columns values from 0 to 1
    # new_val = (old_val - min_val) / (max_val - min_val)
    # cols = data_train.select_dtypes(np.number).columns
    # data_train[cols] = minmax_scale(data_train[cols])
    # data_train = pd.DataFrame(data_train)

    d1 = (data_train - data_train.min())
    d2 = (data_train.max() - data_train.min())
    data_train_scaled = d1 / d2

    # labels = pd.Series(minmax_scale(labels))
    l1 = (labels - labels.min())
    l2 = (labels.max() - labels.min())
    labels_scaled = l1 / l2

    # 5. MODEL TRAINING
    # Python LinearRegression model
    lr_model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, labels_scaled, test_size=0.3, random_state=123, shuffle=False)
    lr_model.fit(X_train, y_train)

    labels_predicted = lr_model.predict(X_test)
    y_pred = pd.Series(data=labels_predicted, name='CO2EMISSIONS_Predicted', index=X_test.index)
    result_dataframe = pd.concat([X_test, y_test, y_pred], axis=1)
    print("LR RESULT DF:")
    print(result_dataframe.head())
    print("===================================")

    # LinearRegressioGradientDescent model
    data_train = data.loc[:, ['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
                       'FUELCONSUMPTION_COMB_MPG']]  # DataFrame
    # data_train = data.loc[:, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]  # DataFrame
    labels = data.loc[:, 'CO2EMISSIONS']  # Series

    ohe = OneHotEncoder(dtype=int, sparse_output=False)
    fueltype = ohe.fit_transform(data_train['FUELTYPE'].to_numpy().reshape(-1, 1))
    data_train = data_train.drop(columns=['FUELTYPE'])
    data_train = data_train.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE'])))

    # Scale all columns values from 0 to 1
    # new_val = (old_val - min_val) / (max_val - min_val)
    # cols = data_train.select_dtypes(np.number).columns
    # data_train[cols] = minmax_scale(data_train[cols])
    # data_train = pd.DataFrame(data_train)
    d1 = (data_train - data_train.min())
    d2 = (data_train.max() - data_train.min())
    data_train_scaled = d1 / d2

    # labels = pd.Series(minmax_scale(labels))
    l1 = (labels - labels.min())
    l2 = (labels.max() - labels.min())
    labels_scaled = l1 / l2

    lrgd_model = LinearRegressionGradientDescent()
    X_train_lrgd, X_test_lrgd, y_train_lrgd, y_test_lrgd = train_test_split(data_train_scaled, labels_scaled, test_size=0.3, random_state=123, shuffle=False)
    lrgd_model.fit(X_train, y_train)

    learning_rates = np.array(
        [[1.0], [0.7], [0.7], [0.875], [1.0], [0.875], [1.0], [0.875], [0.7], [0.7]])

    res_coeff, mse_history = lrgd_model.perform_gradient_descent(learning_rates, 50)

    labels_predicted_lrgd = lrgd_model.predict(X_test_lrgd)
    y_pred_lrgd = pd.Series(data=labels_predicted_lrgd, name='LRGD_CO2EMISSIONS_Predicted', index=X_test_lrgd.index)
    result_dataframe_lrgd = pd.concat([X_test_lrgd, y_test_lrgd, y_pred_lrgd], axis=1)
    print("LRGD RESULT DF:")
    print(result_dataframe_lrgd.head())
    print("===================================")

    # Models statistics

    # MSE
    lrgd_model.set_coefficients(res_coeff)
    print(f'LRGD MSE: {lrgd_model.cost():.6f}')

    c = np.concatenate((np.array([lr_model.intercept_]), lr_model.coef_))
    lrgd_model.set_coefficients(c)
    print(f'LR MSE: {lrgd_model.cost():.6f}')
    print("===================================")
    lrgd_model.set_coefficients(res_coeff)

    # LRGD R2 SCORE
    print(f'LRGD R2 score: {r2_score(y_test_lrgd, labels_predicted_lrgd):.6f}')
    # print(f'LRGD SCORE: {1 - lrgd_model.cost():.6f}')
    # print(f'LRGD COEFF: {lrgd_model.get_coeff()}')

    # LRGD SCORE
    lr_coef_ = lr_model.coef_
    lr_int_ = lr_model.intercept_
    lr_model.coef_ = lrgd_model.coeff.flatten()[1:]
    lr_model.intercept_ = lrgd_model.coeff.flatten()[0]
    print(f'LRGD score: {lr_model.score(X_test_lrgd, y_test_lrgd):.6f}')

    # LR R2 SCORE
    print(f'LR R2 score: {r2_score(y_test, labels_predicted):.6f}')
    # print(f'LR COEFF: {lr_model.coef_}')

    # LR SCORE
    lr_model.coef_ = lr_coef_
    lr_model.intercept_ = lr_int_
    print(f'LR score: {lr_model.score(X_test, y_test):.6f}')
    print("===================================")

    # Vizuelizacija MS_error funkcije kroz iteracije
    # za model koji koristi gradijentni spust.
    plt.figure('MS Error')
    plt.plot(np.arange(0, len(mse_history), 1), mse_history)
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('MS error value', fontsize=13)
    plt.xticks(np.arange(0, len(mse_history), 2))
    plt.title('Mean-square error function')
    plt.legend(['MS Error'])
    plt.show()

    # correlation matrix
    sb.heatmap(data_train.corr(), annot=True, square=True, fmt='.2f')
    plt.show()
