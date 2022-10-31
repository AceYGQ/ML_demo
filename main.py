"""
    Main function of DeepLearning_demo.
    Analyze point cloud by Capon beam-forming.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import HeatMap
import Net


if __name__ == "__main__":
    print("Demo for Heat Map Processing...")

    origin_path = r"E:\Python\Work\data\HeatMap"
    dataset_path = r"dataset"
    testset_path = r"test"
    fname = "103-CPD_103_N_N_N_N_N_N_C1-2"

    # HeatMap.heatmap_visualization(origin_path, fname)
    # HeatMap.heatmap_restore(dataset_path, origin_path, fname)
    # HeatMap.heatmap_restore(testset_path, origin_path, fname)

    train = False
    if train:
        print("Check Device...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        x_train, y_train, x_test, y_test = HeatMap.get_label(dataset_path)
        x_train, x_test = x_train.to(device), x_test.to(device)

        # train
        print("Start Training...")
        net = Net.train(x_train, y_train, device)
        print("\n==========\n")

        # test 1
        print("Start Testing")
        acc_num = 0
        total_num = 0
        for i, data in enumerate(x_test, 0):
            data = data.to(torch.float32)
            prediction = net(data).cpu()
            prediction = int(round(prediction.detach().numpy()[0, 0]))
            # print("Prediction : %.2f\tClass : %d" % (prediction, y_test[i]))
            if prediction == y_test[i]:
                acc_num += 1
            total_num += 1
        print("Accuracy on Test-Set-1 : %.2f\n" % (acc_num / total_num))

        # test 2
        x_train, y_train, x_test, y_test = HeatMap.get_label(testset_path)
        x_train, x_test = x_train.to(device), x_test.to(device)
        # y_train = y_train - 4
        # y_test = y_test - 4

        acc_num = 0
        total_num = 0
        for i, data in enumerate(x_train, 0):
            data = data.to(torch.float32)
            prediction = net(data).cpu()
            prediction = int(round(prediction.detach().numpy()[0, 0]))
            # print("Prediction : %.2f\tClass : %d" % (prediction, y_test[i]))
            if prediction == y_train[i]:
                acc_num += 1
            total_num += 1

        for i, data in enumerate(x_test, 0):
            data = data.to(torch.float32)
            prediction = net(data).cpu()
            prediction = int(round(prediction.detach().numpy()[0, 0]))
            # print("Prediction : %.2f\tClass : %d" % (prediction, y_test[i]))
            if prediction == y_test[i]:
                acc_num += 1
            total_num += 1
        print("Accuracy on Test-Set-2 : %.2f\n" % (acc_num / total_num))

    loss_show = True
    if loss_show:
        loss_name = "loss.txt"
        loss = np.loadtxt(loss_name, delimiter=',')

        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.grid()
        plt.show()
