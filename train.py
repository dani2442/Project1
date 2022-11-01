import torch
import torchvision
from torchvision.transforms import transforms

from models import FashionMLP
from dataset import IN_FEATURES, NUM_CLASSES, get_dataset

L_RATE = 1e-3
GAMMA = 0.9
N_EPOCHS = 5
K=4

# load dataset
train_loader, test_loader = get_dataset()

# select device: cpu vs gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load model
model = FashionMLP(in_size=IN_FEATURES, out_size=NUM_CLASSES)
model.to(device)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=L_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# training
for epoch in range(N_EPOCHS):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass 
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()

    scheduler.step()  
    print("Epoch: {}, Loss: {:.7f}".format(epoch, loss.data))
