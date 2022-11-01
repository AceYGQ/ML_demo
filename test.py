import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import svm

import HeatMap


def PCA_(path):
    # read data & labels
    data, labels = get_data(path)

    # corr_matrix = data.corr(method='pearson')

    # data scaling
    data = scale(data)

    # Analysis Part
    # PCA_analysis(data)

    # rebuild model
    newData = PCA(n_components=3).fit_transform(data)

    # classifier
    clf_svm(newData, labels)


def get_data(path):
    data = []
    labels = []
    imagePaths = sorted(list(HeatMap.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = np.fromfile(imagePath)
        image = np.reshape(image, (-1, 1))
        data.append(image)

        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)

    data = np.array(data)
    data = data.squeeze()
    labels = np.array(labels)

    return data, labels


def PCA_analysis(data):
    pcaModel = PCA(n_components=None)
    pcaModel.fit(data)
    print("===== PCA =====")
    print("Var :\n", pcaModel.explained_variance_)
    print("Ratio :\n", pcaModel.explained_variance_ratio_)


def clf_svm(data, labels):
    clf = svm.SVC(kernel="linear")
    clf.fit(data, labels)

    pre = clf.predict(data).reshape((1, -1)).T
    labels = labels.reshape((1, -1)).T
    # result = np.concatenate((np.array(newData), pre, labels), axis=1)
    result = np.concatenate((pre, labels), axis=1)
    print("prediction :", result)

    count = 0
    for sample in result:
        if sample[-2] != sample[-1]:
            count += 1

    print("mistake number : ", count)
    print("mistake ratio : ", count / result.shape[0])
