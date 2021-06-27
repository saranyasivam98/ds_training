# -*- encoding: UTF-8 -*-

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

__author__ = 'saranya@gyandata.com'

RANGE = range(40, 51)


def cv_predict(model, x, y, k=5):
    return cross_val_predict(model, x, y, cv=KFold(n_splits=k))


def cv_score(model, x, y, k=5, scoring='neg_mean_absolute_error'):
    return cross_val_score(model, x, y, cv=KFold(n_splits=k), scoring=scoring)


def plot_y_y_pred(y, y_pred, k, corr_coeff):
    fig, ax = plt.subplots()
    plt.scatter(y, y_pred)
    plt.axhline(y=y.mean(), color='r', linestyle='-')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.title(f"y vs y_pred for k = {k} with correlation coefficient: {'%.4f'%corr_coeff}")
    plt.show()


def plot_cc(cc):
    fig, ax = plt.subplots()
    plt.plot(RANGE, cc)
    plt.xlabel("No of folds")
    plt.ylabel("Correlation Coefficient")
    plt.title("No of folds vs Correlation Coefficient RFR")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.show()


def main():
    cc_list = []
    print("reached main")
    for k in RANGE:
        x, y = make_regression(n_samples=50, n_features=2)
        print(f"Reached {k}th iteration")
        regressor = RandomForestRegressor()

        y_pred = cv_predict(regressor, x, y, k)
        # scores = cv_score(regressor, x, y, k)

        y_pred = y_pred.reshape(50, 1)
        y = y.reshape(50, 1)

        var = np.concatenate((y, y_pred), axis=1)
        corr_coeff = np.corrcoef(var.T)[0, 1]
        cc_list.append(corr_coeff)

    plot_cc(cc_list)


if __name__ == '__main__':
    main()
