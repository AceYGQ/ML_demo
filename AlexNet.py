import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # first convolution layer
    nn.Conv2d(27, 16, kernel_size=5, stride=1, padding=0), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1),

    # second convolution layer
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1),

    # third, forth and fifth convolution layer
    # nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    # nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    # nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    # nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),

    # three full connecting layer, the last is output
    nn.Linear(3200, 512), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 64), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(64, 1)
)

# output dimension
X = torch.randn(1, 27, 28, 28)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# import data and pre-processing
# batch_size = 128
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
#
# lr, num_epochs = 0.01, 10
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
