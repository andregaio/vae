from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


def load_dataset(batch_size):

    train_dataset = MNIST(root='./data', train=True, transform=TRANSFORMS,download=True)
    val_dataset = MNIST(root='./data', train=False, transform=TRANSFORMS, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader  = DataLoader(dataset=val_dataset,  batch_size=batch_size, 
                             shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader