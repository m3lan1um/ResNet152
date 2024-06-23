import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from model import Bottleneck, ResNet

import json

from accelerate import Accelerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocessing
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
# dataset
root = 'datasets/'
train_ds = datasets.CIFAR10(root, train=True, transform=transform_train, download=True)
test_ds = datasets.CIFAR10(root, train=False, transform=transform_test, download=True)
train_ds, val_ds = torch.utils.data.random_split(train_ds, [45_000, 5_000])

# dataset size check
print(f'train dataset size: {len(train_ds)}')   # 45k
print(f'val dataset size: {len(val_ds)}')       # 5k
print(f'test dataset size: {len(test_ds)}')     # 10k

# model 
num_layers = [3, 8, 36, 3]
model = ResNet(ResBlock=Bottleneck, num_layers=num_layers, num_classes=10).to(device)
# loss function
loss_func = nn.CrossEntropyLoss().to(device)
# optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=5e-5, patience=10, verbose=True) 
# dataloader
batch_size = 128
num_workers = 4
dataloader_train = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataloader_val = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
dataloader_test = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

# train & validation
epoch = 100
best_val_loss = float('inf')
best_val_acc = 0

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
for e in range(epoch):
    print(f'===== epoch: {e + 1} / {epoch} =====')

    # train
    train_loss, train_acc = 0, 0
    model.train()
    for idx, (X, y) in enumerate(dataloader_train):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    train_loss /= len(dataloader_train)
    train_acc /= len(dataloader_train.dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f'Epoch: {e + 1}\tTrain loss: {train_loss} -- Train accuracy: {train_acc}')
    
    # validation
    val_loss, val_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader_val:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_func(pred, y).item()
            val_acc += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        val_loss /= len(dataloader_val)
        val_acc /= len(dataloader_val.dataset)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f'Epoch: {e + 1}\tVal loss: {val_loss} -- Val accuracy: {val_acc}')

    # save checkpoint: best accuracy
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': e + 1,
            'val_loss': val_loss
        }
        torch.save(checkpoint, f'checkpoints/ResNet152_final_acc_{val_acc}.pth')
        print(f'Checkpoint saved at epoch {e + 1}')

results = {
    'train_losses': train_loss_list,
    'train_accuracies': train_acc_list,
    'val_losses': val_loss_list,
    'val_accuracies': val_acc_list
}

with open('results/training_results.json', 'w') as f:
    json.dump(results, f)
