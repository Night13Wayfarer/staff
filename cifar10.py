import numpy as np
import torch

from tqdm import tqdm

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


MODEL_PATH = './model_cifar.pt'


def create_loaders(batch_size=32, val_size=0.2):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_data = datasets.CIFAR10(
        'data', train=True, download=True, transform=train_transform
    )
    val_data = datasets.CIFAR10(
        'data', train=True, download=True, transform=test_transform
    )
    test_data = datasets.CIFAR10(
        'data', train=False, download=True, transform=test_transform
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, sampler=val_sampler
    )
    test_loader = DataLoader(test_data, batch_size=batch_size)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return loaders


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    return train_loss / len(data_loader.dataset)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0.0
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item() * images.size(0)
    return val_loss / len(data_loader.dataset)


def train_model(model, loaders, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 150
    val_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, criterion, optimizer, loaders['train'], device
        )
        val_loss = evaluate(model, criterion, loaders['val'], device)
        print('Epoch: {}\tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(
            epoch, train_loss, val_loss)
        )
        if val_loss <= val_loss_min:
            print('Saving best model ...')
            torch.save(model.state_dict(), MODEL_PATH)
            val_loss_min = val_loss


def test_model(model, data_loader, device):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    n_correct = 0
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.reshape(*top_class.shape)
            n_correct += torch.sum(equals.type(torch.IntTensor)).item()
    print('Test accuracy: {:.2f} %'.format(
        100.0 * n_correct / len(data_loader.dataset)
    ))


def main():
    loaders = create_loaders()
    criterion = nn.NLLLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network()
    model.to(device)

    train_model(model, loaders, device)
    test_model(model, loaders['test'], device)


main()
