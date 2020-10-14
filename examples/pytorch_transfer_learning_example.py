import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchvision.models import *

model = alexnet(pretrained=True)


transform = transforms.Compose(
    [ transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    from cnn_raccoon import inspector
    inspector(model, images, 10, engine="pytorch",)
