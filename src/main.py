import sys

import torch

import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
nets = {
    "lenet": trainer.train_lenet,
    "alexnet": trainer.train_alexnet
}

if __name__ == '__main__':
    net_name = sys.argv[1].lower()
    if net_name not in nets.keys():
        raise NotImplementedError(f"{net_name} not implemented.")

    print(f"train {net_name} on {device}")
    nets[net_name](device)
