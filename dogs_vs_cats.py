import os
import random
import shutil
import torch

from collections import OrderedDict
from tqdm import tqdm

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


INPUT_PATHS = {
    'train': './dogs-vs-cats/train',
    'test': './dogs-vs-cats/test'
}

DATA_PATHS = {
    'train': './train',
    'val': './val',
    'test': './test'
}

FIRST_STAGE_MODEL = 'first_stage.pth'
SECOND_STAGE_MODEL = 'second_stage.pth'


def create_data_folders():
    for key, path in DATA_PATHS.items():
        os.mkdir(path)
        if key != 'test':
            for label in ('cat', 'dog'):
                os.mkdir(os.path.join(path, label))


def delete_data_folders():
    for path in DATA_PATHS.values():
        if os.path.exists(path):
            shutil.rmtree(path)


def make_data_folders(train_ratio=0.8, random_seed=42):
    delete_data_folders()
    create_data_folders()

    random.seed(random_seed)
    train_images = os.listdir(INPUT_PATHS['train'])
    random.shuffle(train_images)
    train_size = int(len(train_images) * train_ratio)

    for i, image in enumerate(tqdm(train_images)):
        path = DATA_PATHS['train'] if i < train_size else DATA_PATHS['val']
        label = 'cat' if image.startswith('cat') else 'dog'
        name = image[4:]
        shutil.copy(
            os.path.join(INPUT_PATHS['train'], image),
            os.path.join(path, label, name)
        )

    test_images = os.listdir(INPUT_PATHS['test'])
    for image in tqdm(test_images):
        shutil.copy(
            os.path.join(INPUT_PATHS['test'], image),
            os.path.join(DATA_PATHS['test'], image)
        )


def create_model():
    model = models.mnasnet1_0(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1280, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model


class TestDataSet(Dataset):
    def __init__(self, img_folder, transform):
        self.img_folder = img_folder
        self.transform = transform
        image_ids = [int(f_name.split('.')[0]) for f_name in os.listdir(img_folder)]
        self.images = ['{}.jpg'.format(image_id) for image_id in sorted(image_ids)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_folder, self.images[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image


def create_loaders(batch_size=64):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(DATA_PATHS['train'], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(DATA_PATHS['val'], transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TestDataSet(DATA_PATHS['test'], test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return loaders


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        logps = model(images)
        batch_loss = criterion(logps, labels)
        train_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
    return train_loss / len(data_loader)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logps = model(images)
            batch_loss = criterion(logps, labels)
            total_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.reshape(*top_class.shape)
            total_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return total_loss / len(data_loader), total_accuracy / len(data_loader)


def train_classifier_layer(model, loaders, criterion, device):
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 2
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, loaders['train'], device)
        val_loss, val_accuracy = evaluate(model, criterion, loaders['val'], device)
        print('Epoch {}/{}    train_loss: {}     val_loss: {}    val_accuracy: {}'.format(
            epoch + 1, epochs, train_loss, val_loss, val_accuracy
        ))
    torch.save(model.state_dict(), FIRST_STAGE_MODEL)


def train_all_layers(model, loaders, criterion, device):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    best_acc = 0.0
    best_state = None

    epochs = 10
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, loaders['train'], device)
        val_loss, val_accuracy = evaluate(model, criterion, loaders['val'], device)
        print('Epoch {}/{}    train_loss: {}     val_loss: {}    val_accuracy: {}'.format(
            epoch + 1, epochs, train_loss, val_loss, val_accuracy
        ))
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_state = model.state_dict()  # TODO: use deepcopy
    model.load_state_dict(best_state)
    torch.save(best_state, SECOND_STAGE_MODEL)


def predict_test(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(data_loader):
            images = images.to(device)
            logps = model(images)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            labels = [elem[0].item() for elem in top_class]
            predictions.extend(labels)
    return predictions


def write_to_file(output_file, predictions):
    with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i + 1, label))


def main():
    make_data_folders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model.to(device)

    loaders = create_loaders()
    criterion = nn.NLLLoss()

    train_classifier_layer(model, loaders, criterion, device)
    train_all_layers(model, loaders, criterion, device)

    predictions = predict_test(model, loaders['test'], device)
    write_to_file('result.csv', predictions)

    delete_data_folders()


main()
