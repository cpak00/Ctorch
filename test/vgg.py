import torch
import torch.nn as nn
import torchvision

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print(x.max())
        x = self.conv1(x)
        print(x.max())
        x = self.conv2(x)
        print(x.max())
        x = self.pool1(x)
        print(x.max())
        x = self.conv3(x)
        print(x.max())
        x = self.conv4(x)
        print(x.max())
        x = self.pool2(x)
        print(x.max())
        x = self.conv5(x)
        print(x.max())
        x = self.conv6(x)
        print(x.max())
        x = self.pool3(x)
        print(x.max())

        print(x.shape)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/share/dataset/MNIST', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor(),
                                    #torchvision.transforms.Normalize(
                                    #    (0.1307,), (0.3081,))
                                    #
                                ])),
    batch_size=10, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/share/dataset/MNIST', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor(),
                                    #torchvision.transforms.Normalize(
                                    #    (0.1307,), (0.3081,))
                                    #
                                ])),
    batch_size=10, shuffle=True)

    data, label = train_loader.__iter__().__next__()

    model = VGG()
    output = model(data)
    print(output)