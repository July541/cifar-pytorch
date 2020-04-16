from time import time

import torch
import torchvision
from torch.utils.data import DataLoader


def load_data(batch_size=256, resize=None):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))  # for alexnet
    trans.append(torchvision.transforms.ToTensor())
    transformer = torchvision.transforms.Compose(trans)

    train_set = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transformer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transformer)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(data_iter, net: torch.nn.Module, device=None):
    if device is None:
        device = list(net.parameters())[0].device

    acc, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # forbid dropout
            acc += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()

            n += y.shape[0]

    return acc / n


def train(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    train_l_sum, train_acc_sum, n, batch_count, t1 = 0.0, 0.0, 0, 0, time()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        if device == torch.device("cpu") or (epoch + 1) % 10 == 0:
            test_acc = evaluate(test_iter, net)
            print("epoch {}, loss {:.4f}, train_acc {:.4f}, test_acc {:.4f}, time {:.1f} sec"
                  .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time() - t1))

            train_l_sum, train_acc_sum, n, batch_count, t1 = 0.0, 0.0, 0, 0, time()
