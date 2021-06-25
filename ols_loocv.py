# -*- encoding: UTF-8 -*-

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, KFold
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter

__author__ = 'saranya@gyandata.com'

RANGE = range(3, 5)


def generate_xy():
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    return x, y


def cv_predict(model, x, y, k):
    return cross_val_predict(model, x, y, cv=KFold(n_splits=k))


def cv_score(model, x, y, k, scoring='neg_mean_absolute_error'):
    return cross_val_score(model, x, y, cv=KFold(n_splits=k), scoring=scoring)


def plot_y_y_pred(y, y_pred, k, corr_coeff):
    fig, ax = plt.subplots()
    plt.scatter(y, y_pred)
    plt.axhline(y=y.mean(), color='r', linestyle='-')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.title(f"y vs y_pred for k = {k} with correlation coefficient: {'%.4f'%corr_coeff}")
    plt.show()


def plot_cc(cc):
    fig, ax = plt.subplots()
    plt.plot(RANGE, cc)
    plt.xlabel("No of folds")
    plt.ylabel("Correlation Coefficient")
    plt.title("No of folds vs Correlation Coefficient")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.show()


def main():
    scores_dict = {}
    cc_list = []
    for k in RANGE:
        x, y = generate_xy()
        dummy_regr = DummyRegressor(strategy="mean")
        y_pred = cv_predict(dummy_regr, x, y, k)

        scores = cv_score(dummy_regr, x, y, k)
        scores_dict[k] = scores

        y_pred = y_pred.reshape(50, 1)
        var = np.concatenate((y, y_pred), axis=1)
        corr_coeff = np.corrcoef(var.T)[0, 1]
        plot_y_y_pred(y, y_pred, k, corr_coeff)
        cc_list.append(corr_coeff)

    plot_cc(cc_list)


if __name__ == '__main__':
    main()
