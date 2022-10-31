import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.datasets import load_iris

# 读取处理数据
# originalData = pd.read_csv()
iris = load_iris()
originalData = iris.data
label = iris.target

# 查看相关系数矩阵
# corr_matrix = originalData.corr(method='pearson')

# 数据标准化
data = scale(originalData)

# 使用sklearn的主成分分析，用于判断保留主成分的数量
pcaModel = PCA(n_components=None)
pcaModel.fit(data)
print(pcaModel.explained_variance_)
print(pcaModel.explained_variance_ratio_)

# 重新建模
newData = PCA(n_components=3).fit_transform(data)
# originalData['score'] = newData
# originalData.sort_values('score', ascending=False)
# print(newData)

clf = svm.SVC(kernel="linear")
clf.fit(newData, label)

pre = clf.predict(newData).reshape((1, -1)).T
label = label.reshape((1, -1)).T
result = np.concatenate((np.array(newData), pre, label), axis=1)
# print("prediction :", result)

count = 0
for sample in result:
    if sample[-2] != sample[-1]:
        count += 1

print("mistake number : ", count)
print("mistake ratio : ", count / result.shape[0])
