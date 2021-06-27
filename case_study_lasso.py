# -*- encoding: UTF-8 -*-

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, KFold
from sklearn import linear_model
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
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4e'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4e'))
    plt.title(f"y vs y_pred for k = {k} with correlation coefficient: {'%.4f'%corr_coeff}")
    plt.show()


def plot_cc(cc):
    fig, ax = plt.subplots()
    plt.plot(RANGE, cc)
    plt.xlabel("No of folds")
    plt.ylabel("Correlation Coefficient")
    plt.title("No of folds vs Correlation Coefficient Lasso model")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3e'))
    plt.show()


def main():
    cc_list = []
    for k in RANGE:
        x, y = make_regression(n_samples=50, n_features=2)
        regressor = linear_model.Lasso(alpha=0.1)

        y_pred = cv_predict(regressor, x, y, k)
        # scores = cv_score(regressor, x, y, k)

        y_pred = y_pred.reshape(50, 1)
        y = y.reshape(50, 1)

        var = np.concatenate((y, y_pred), axis=1)
        corr_coeff = np.corrcoef(var.T)[0, 1]
        cc_list.append(corr_coeff)

    plot_cc(cc_list)
    print(cc_list)


if __name__ == '__main__':
    main()
