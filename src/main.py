import torch
from lenet import LeNet
from utils import train, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"train on {device}")


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    train_iter, test_iter = load_data()
    net = LeNet()
    lr = 0.001
    epoch_num = 120
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, optimizer, device, epoch_num)


if __name__ == '__main__':
    main()
