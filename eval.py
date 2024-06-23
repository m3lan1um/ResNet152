import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets

from model import Bottleneck, ResNet


dir = 'datasets/'
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
dataset_test = datasets.CIFAR10(dir, train=False, transform=transform_test, download=True)

batch_size = 512
num_workers = 4
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)

num_layers = [3, 8, 36, 3] 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet(ResBlock=Bottleneck, num_layers=num_layers, num_classes=10).to(device)

checkpoint = "checkpoints/ResNet152_final_acc_0.926.pth"
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

loss_func = nn.CrossEntropyLoss().to(device)

loss, acc = 0, 0
model.eval()
with torch.no_grad():
    for idx, (X, y) in enumerate(dataloader_test):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss += loss_func(pred, y).item()
        acc += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    loss /= len(dataloader_test)
    acc /= len(dataloader_test.dataset)

print(f'Test loss: {loss} -- Test accuracy: {acc}')
