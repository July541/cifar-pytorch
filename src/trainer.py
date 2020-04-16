import torch
from lenet import LeNet
from alexnet import AlexNet
from utils import train, load_data


def run(net, lr, train_iter, test_iter, device, epoch_num=500):
    lr = lr
    epoch_num = epoch_num
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, optimizer, device, epoch_num)


def train_lenet(device):
    lr = 0.001
    train_iter, test_iter = load_data()
    net = LeNet()
    epoch_num = 200
    run(net, lr, train_iter, test_iter, device, epoch_num)


def train_alexnet(device):
    lr = 0.01
    train_iter, test_iter = load_data(batch_size=128, resize=227)
    net = AlexNet()
    run(net, lr, train_iter, test_iter, device)
