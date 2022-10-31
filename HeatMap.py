import numpy as np
import numpy.linalg as la
import os
# import matplotlib
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# matplotlib.use('TkAgg')


def normalization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges

    return normData


def list_images(basePath, contains=None):
    # 返回有效的图片路径数据集
    return list_files(basePath, validExts='.bin', contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # 遍历图片数据目录，生成每张图片的路径
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环遍历当前目录中的文件名
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # 通过确定.的位置，从而确定当前文件的文件扩展名
            ext = filename[filename.rfind("."):].lower()

            # 检查文件是否为图像，是否应进行处理
            if validExts is None or ext.endswith(validExts):
                # 构造图像路径
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def draw(top, front, left, slice_show):
    # plt.clf()
    plt.grid()

    plt.subplot(1, 3, 1)
    pd_top = pd.DataFrame(top)
    sns.heatmap(data=pd_top,
                cmap='brg_r',
                linewidths=.1,
                annot=False,
                fmt='.1e',
                vmax=None,
                vmin=None
                )
    plt.title("Slice %d" % slice_show[0])

    plt.subplot(1, 3, 2)
    pd_front = pd.DataFrame(front)
    sns.heatmap(data=pd_front,
                cmap='brg_r',
                linewidths=.1,
                annot=False,
                fmt='.1e',
                vmax=None,
                vmin=None
                )
    plt.title("Slice %d" % slice_show[1])

    plt.subplot(1, 3, 3)
    pd_left = pd.DataFrame(left)
    sns.heatmap(data=pd_left,
                cmap='brg_r',
                linewidths=.1,
                annot=False,
                fmt='.1e',
                vmax=None,
                vmin=None
                )
    plt.title("Slice %d" % slice_show[2])

    plt.show()
    plt.pause(0.1)


def heatmap_visualization(path, fname):
    print("Heat Map Visualization.\nStart Reading File...")
    heatmap = np.loadtxt(path + '\\' + fname + ".txt", delimiter=',')
    print("Reading Complete.")

    plt.ion()
    plt.figure(figsize=(18, 6))
    # mngr = plt.get_current_fig_manager()  # 获取当前figure manager
    # mngr.window.wm_geometry("+50+50")  # 调整窗口在屏幕上弹出的位置

    frame_num = heatmap[:, 0].max()
    index = int(heatmap[:, 1].max()) + 1
    azimuth = int(heatmap[:, 2].max()) + 1
    elevation = int(heatmap[:, 3].max()) + 1
    slice_show = [15, 17, 19]
    for i in range(int(frame_num)):
        data = heatmap[heatmap[:, 0] == i + 1, 1:]
        slice_1 = np.zeros((azimuth, elevation))
        slice_2 = np.zeros((azimuth, elevation))
        slice_3 = np.zeros((azimuth, elevation))
        for point in data:
            if point[0] == slice_show[0]:
                slice_1[int(point[1]), int(point[2])] = point[3]
            if point[0] == slice_show[1]:
                slice_2[int(point[1]), int(point[2])] = point[3]
            if point[0] == slice_show[2]:
                slice_3[int(point[1]), int(point[2])] = point[3]

        draw(slice_1, slice_2, slice_3, slice_show)
        plt.clf()


def heatmap_restore(addr, path, fname):
    print("Heat Map Restoration.\nStart Reading File...")
    heatmap = np.loadtxt(path + '\\' + fname + ".txt", delimiter=',')
    print("Reading Complete.")

    output_addr = addr
    if not os.path.exists(output_addr):
        os.mkdir(output_addr)

    output_addr = os.path.join(output_addr, fname)
    if not os.path.exists(output_addr):
        os.makedirs(output_addr)

    frame_num = int(heatmap[:, 0].max())
    for i in range(frame_num):
        data = heatmap[heatmap[:, 0] == i + 1, 1:]
        file_name = fname + "_%d.bin" % (i + 1)

        tmp = np.zeros((27, 28, 28))
        for point in data:
            tmp[int(point[0]) - 4, int(point[1]), int(point[2])] = point[3]

        tmp.tofile(os.path.join(output_addr, file_name))


def get_label(path):
    # 加载自建数据集
    data = []
    labels = []

    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)

    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据
        image = np.fromfile(imagePath)
        image = np.reshape(image, (27, 28, 28))
        image = normalization(image)
        data.append(image)
        # 读取标签
        label = int(imagePath.split(os.path.sep)[-2])  # 文件路径的倒数第二个就是文件夹的名字被定义为标签
        # labels.append(label)
        label_vector = np.zeros((1, 4))
        label_vector[0, label - 1] = 1
        labels.append(label_vector)

    data = np.array(data)
    labels = np.array(labels)
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

    x_train = torch.from_numpy(x_train).to(torch.float32)
    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    # y_test = torch.from_numpy(y_test).to(torch.float32)

    return x_train, y_train, x_test, y_test
