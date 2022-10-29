#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

class FashionCNN(nn.Module):   
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.ReLU()
        )
#        self.drop = nn.Dropout(0.25)
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=600, out_features=10),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(in_features=10, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
#        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

imsizesq = 28
num_classes = 10
batch = 100
num_epochs = 5
learning_rate = 0.001

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
            transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
            transforms.Compose([transforms.ToTensor()]))  
                                            
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch)
model = FashionCNN()
model.to(device)
model.fc2.register_forward_hook(get_activation('fc2'))

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
    
        train = images.view(batch, 1, imsizesq, imsizesq)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
        
    print("Epoch: {}, Loss: {:.7f}".format(epoch, loss.data))



class_correct = [0. for _ in range(num_classes)]
total_correct = [0. for _ in range(num_classes)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test = images
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()
        
        for i in range(batch):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1

# Saving the fc2 layer outputs
images_train = []
output_train_fc2 = []
for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        train = images.view(batch, 1, imsizesq, imsizesq)
        outputs = model(train)
        images_train.append(images)
        output_train_fc2.append(activation["fc2"])

print("Test Set Accuracy")      
for i in range(num_classes):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))



