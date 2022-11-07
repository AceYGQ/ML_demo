"""
    Main function of DeepLearning_demo.
    Analyze point cloud by Capon beam-forming.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import HeatMap
import PCA
import Net


if __name__ == "__main__":
    print("Demo for Heat Map Processing...")

    origin_path = r"E:\Python\Work\data\HeatMap"
    dataset_path = r"dataset"
    testset_path = r"test"
    fname = "7-Localization_7_N_N_N_N_N_N_A-1"

    # HeatMap.heatmap_visualization(origin_path, fname)
    # HeatMap.heatmap_restore(dataset_path, origin_path, fname)
    # HeatMap.heatmap_restore(testset_path, origin_path, fname)

    # ML Part
    PCA.PCA_CLF(dataset_path)

    # NN Part
    train = False
    if train:
        print("Check Device...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        x_train, y_train, x_test, y_test = HeatMap.get_label(dataset_path)
        x_train = torch.from_numpy(x_train).to(torch.float32)
        x_test = torch.from_numpy(x_test).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(torch.float32)
        x_train, x_test = x_train.to(device), x_test.to(device)

        # train
        print("Start Training on %d samples" % x_train.shape[0])
        net = Net.train(x_train, y_train, device)
        print("\n==========\n")

        # test 1
        print("Start Testing on %d samples" % x_test.shape[0])
        acc_num = 0
        total_num = 0
        for i, data in enumerate(x_test, 0):
            prediction = net(data).cpu()
            prediction = prediction.detach().numpy()
            if np.argmax(prediction) == np.argmax(y_test[i]):
                acc_num += 1
            total_num += 1
        print("Accuracy on Test-Set-1 : %.2f\n" % (acc_num / total_num))

        # test 2
        test_2 = False
        if test_2:
            x_train, y_train, x_test, y_test = HeatMap.get_label(testset_path)
            x_train = torch.from_numpy(x_train).to(torch.float32)
            x_test = torch.from_numpy(x_test).to(torch.float32)
            x_train, x_test = x_train.to(device), x_test.to(device)

            acc_num = 0
            total_num = 0
            for i, data in enumerate(x_train, 0):
                prediction = net(data).cpu()
                prediction = prediction.detach().numpy()
                if np.argmax(prediction) == np.argmax(y_train[i]):
                    acc_num += 1
                total_num += 1

            for i, data in enumerate(x_test, 0):
                prediction = net(data).cpu()
                prediction = prediction.detach().numpy()
                if np.argmax(prediction) == np.argmax(y_test[i]):
                    acc_num += 1
                total_num += 1
            print("Accuracy on Test-Set-2 : %.2f\n" % (acc_num / total_num))

    loss_show = False
    if loss_show:
        loss_name = "loss.txt"
        loss = np.loadtxt(loss_name, delimiter=',')

        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.grid()
        plt.show()
