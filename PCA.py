import numpy as np
# import pandas as pd
import os
import time
import random
# import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import HeatMap


def PCA_CLF(path):
    print("========== PCA ==========\n")
    # read data & labels
    data, labels = get_data(path)

    # tmp = data[0, :].reshape((1, -1))
    # corr_matrix = pd.DataFrame(tmp).corr(method='pearson')
    # print(corr_matrix)

    # data scaling
    data = scale(data)

    # Analysis Part
    # PCA_analysis(data)

    # rebuild model
    newData = PCA(n_components=4).fit_transform(data)

    # classifier
    # clf_svm(newData, labels)
    # clf_knn(newData, labels)
    clf_rf(newData, labels)


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
    print("Var :\n", pcaModel.explained_variance_)
    print("Ratio :\n", pcaModel.explained_variance_ratio_)


def clf_svm(data, labels):
    print("***** classifier : SVM *****")
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

    # clf = svm.SVC(kernel="linear")
    # clf = svm.SVC(kernel="poly")
    # clf = svm.SVC(kernel="sigmoid")
    clf = svm.SVC(kernel="rbf")
    start_time = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time()

    pre = clf.predict(x_test).reshape((-1, 1))
    predict_time = time.time()

    y_test = y_test.reshape((-1, 1))
    result = np.concatenate((pre, y_test), axis=1)
    # print("prediction :", result)
    print("train time : %.4fs , predict time : %.4fs" % (train_time - start_time, predict_time - train_time))

    # Pearson correlation coefficient & p
    # R2 - coefficient of determination: the closer to 1,  the better the model is.
    # RMSE - Root Mean Square Error
    pearson_r = stats.pearsonr(y_test.squeeze(), pre.squeeze())
    R2 = metrics.r2_score(y_test, pre)
    RMSE = metrics.mean_squared_error(y_test, pre) ** 0.5
    print('Pearson correlation coefficient is {0}, R2 is {1} and RMSE is {2}.'.format(pearson_r, R2, RMSE))

    count = 0
    for sample in result:
        if sample[-2] != sample[-1]:
            count += 1
    print("mistake number : ", count)
    print("mistake ratio : %.4f\n" % (count / result.shape[0]))


def clf_knn(data, labels):
    print("***** classifier : KNN *****")
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)

    start_time = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time()
    pre = clf.predict(x_test).reshape((-1, 1))
    predict_time = time.time()

    y_test = y_test.reshape((-1, 1))
    result = np.concatenate((pre, y_test), axis=1)
    # print("prediction :", result)
    print("train time : %.4fs , predict time : %.4fs" % (train_time - start_time, predict_time - train_time))

    # Pearson correlation coefficient & p
    # R2 - coefficient of determination: the closer to 1,  the better the model is.
    # RMSE - Root Mean Square Error
    pearson_r = stats.pearsonr(y_test.squeeze(), pre.squeeze())
    R2 = metrics.r2_score(y_test, pre)
    RMSE = metrics.mean_squared_error(y_test, pre) ** 0.5
    print('Pearson correlation coefficient is {0}, R2 is {1} and RMSE is {2}.'.format(pearson_r, R2, RMSE))

    count = 0
    for sample in result:
        if sample[-2] != sample[-1]:
            count += 1
    print("mistake number : ", count)
    print("mistake ratio : %.4f\n" % (count / result.shape[0]))


def clf_rf(data, labels):
    print("***** classifier : RF *****")

    # pre-set parameters
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)

    start_time = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time()
    pre = clf.predict(x_test).reshape((-1, 1))
    predict_time = time.time()

    y_test = y_test.reshape((-1, 1))
    result = np.concatenate((pre, y_test), axis=1)
    # print("prediction :", result)
    print("train time : %.4fs , predict time : %.4fs" % (train_time - start_time, predict_time - train_time))

    # Pearson correlation coefficient & p
    # R2 - coefficient of determination: the closer to 1,  the better the model is.
    # RMSE - Root Mean Square Error
    pearson_r = stats.pearsonr(y_test.squeeze(), pre.squeeze())
    R2 = metrics.r2_score(y_test, pre)
    RMSE = metrics.mean_squared_error(y_test, pre) ** 0.5
    print('Pearson correlation coefficient is {0}, R2 is {1} and RMSE is {2}.'.format(pearson_r, R2, RMSE))

    count = 0
    for sample in result:
        if sample[-2] != sample[-1]:
            count += 1
    print("mistake number : ", count)
    print("mistake ratio : %.4f\n" % (count / result.shape[0]))

    # k-cross validate
    # cv = cross_validate(clf, data, labels, cv=5, scoring='accuracy', return_estimator=True)
    # scores = cross_val_score(clf, data, labels, cv=5, scoring='accuracy').mean()
    # print(scores)

    # grid search and k-cross validate
    # param_grid = [
    #     {'n_estimators': range(10, 1001, 10), 'max_features': ['auto', 'sqrt', 'log2']},
    # ]
    # clf = RandomForestClassifier()
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(data, labels)
    # print(grid_search.best_params_)
