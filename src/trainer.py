import torch
from lenet import LeNet
from alexnet import AlexNet
from utils import train, load_data


def run(net, train_iter, test_iter, device, epoch_num=500):
    lr = 0.001
    epoch_num = epoch_num
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, optimizer, device, epoch_num)


def train_lenet(device):
    train_iter, test_iter = load_data()
    net = LeNet()
    epoch_num = 200
    run(net, train_iter, test_iter, device, epoch_num)


def train_alexnet(device):
    train_iter, test_iter = load_data(batch_size=128, resize=227)
    net = AlexNet()
    run(net, train_iter, test_iter, device)