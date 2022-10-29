import torch
import torchvision
from torchvision.transforms import transforms

BATCH_SIZE = 32

from models import FashionMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
            transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
            transforms.Compose([transforms.ToTensor()]))  
                                            
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
model = FashionMLP()
model.to(device)
