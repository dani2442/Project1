import torch
from torch import nn
from torch.nn import Sequential, ReLU, Tanh, Linear

from utils.dataset import NUM_CLASSES


class FashionMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(FashionMLP, self).__init__()

        self.mlp = Sequential(
            Linear(in_size, 10), ReLU(),
            Linear(10, out_size)
        )

    def forward(self, x):
        x = x.flatten(1,3)
        return self.mlp(x)


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
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=600, out_features=10),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(in_features=10, out_features=10)

        self.feature_extraction = nn.Sequential(
            self.layer1,
            self.layer2
        )
        self.classifier = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )
        
    def forward(self, x):
        out = self.feature_extraction(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class FashionSVM(nn.Module):   
    def __init__(self, feature_extractor, stop_layer, get_input=False):
        super(FashionSVM, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.stop_layer = stop_layer

        getattr(self.feature_extractor,stop_layer).register_forward_hook(self._get_activation(stop_layer, get_input))
        self._activation = None

        self.svm = nn.LazyLinear(out_features=NUM_CLASSES)
        
    def forward(self, x):
        self.feature_extractor(x)
        out = self._activation.flatten(1).requires_grad_(False)
        out = self.svm(out)

        return out


    def _get_activation(self, name, get_input=False):
        if get_input:
            def hook(model, input, output):
                self._activation = input.detach()
        else:
            def hook(model, input, output):
                self._activation = output.detach()
        return hook



class LeastSquaresClassifier():
    def __init__(self, lambd=0.1):
        self.lambd = lambd

    def fit(self, X, y):
        _, f = X.shape

        A = torch.sum(
                torch.matmul(
                    torch.unsqueeze(X,dim=-1), 
                    torch.unsqueeze(X, dim=-2)
                ),
                dim=0
            ) + (
                self.lambd * torch.eye(f)
            )

        Y = 2.*nn.functional.one_hot(y, num_classes = NUM_CLASSES)-1
        B = torch.sum(
            torch.matmul(
                torch.unsqueeze(X, dim=-1),
                torch.unsqueeze(Y, dim=-2)
            ),
            dim=0
        )

        self.W = torch.matmul(torch.inverse(A), B)

    def predict(self, X):
        return torch.argmax(torch.matmul(self.W.T, X.T), dim=0)

    def score(self, X, y):
        b, f = X.shape
        return (torch.sum(self.predict(X)==y)/b).item()

    def loss(self, X, y):
        A = nn.functional.one_hot(y, num_classes = NUM_CLASSES) - torch.matmul(self.W.T, X.T)
        B = self.lambd * torch.matmul(self.W.T, self.W).trace()

        return torch.sum(A, dim=0) + B


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, labels):
        loss = torch.tensor(0.)
        for i, l in enumerate(labels):
            for j in range(NUM_CLASSES):
                if j==l: continue
                loss+= torch.max(torch.tensor(0.), torch.tensor(1.)-(output[i,l]-output[i,j]))
        return loss/len(labels)
