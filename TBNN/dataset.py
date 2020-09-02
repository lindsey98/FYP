from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def minst_data_loader_test(batch_size):
    test_dataset = MNIST('./data', transform=img_transform, train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def minst_data_loader_train(batch_size):
    train_dataset = MNIST('./data', transform=img_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
