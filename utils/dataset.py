import torch
import torchvision
import torchvision.transforms as T

classes = ("T-shirt/Top", "Trouser", "Pullover", "Dress",
 "Coat",  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

IN_FEATURES = 28 * 28
NUM_CLASSES = len(classes)

def get_dataset():
    t = T.Compose([T.ToTensor()])

    train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=t)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=t)  

    print("Train/Test: {:.0f}%/{:.0f}%".format(
        100*len(train_set)/(len(test_set)+len(train_set)),
        100*len(test_set)/(len(test_set)+len(train_set))
    ))

    return train_set, test_set

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader
