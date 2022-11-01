# 用python实现主成分分析（PCA）
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.datasets import load_iris


def pca(X, k):
    X_decentralization = X - X.mean(axis=0)             # 向量X去中心化
    X_cov = np.cov(X_decentralization.T, ddof=0)        # 计算向量X的协方差矩阵，自由度可以选择0或1
    eigen_values, eigen_vectors = eig(X_cov)              # 计算协方差矩阵的特征值和特征向量
    k_large_index = eigen_values.argsort()[-k:][::-1]    # 选取最大的K个特征值及其特征向量
    k_eigenvectors = eigen_vectors[k_large_index]        # 用X与特征向量相乘
    return np.dot(X_decentralization, k_eigenvectors.T)


if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    # num = 2
    # data_pca = pca(data, num)
    # print(data_pca)

    # 计算协方差矩阵
    data_cov = np.cov(data.T, ddof=0)

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = eig(data_cov)

    tot = sum(eigenvalues)
    var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 5), var_exp, alpha=0.5, align='center', label='individual var')
    plt.step(range(1, 5), cum_var_exp, where='mid', label='cumulative var')
    plt.ylabel('variance ration')
    plt.xlabel('principal components')
    plt.legend(loc='best')
    plt.show()
