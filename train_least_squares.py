import numpy as np
import argparse
import pickle

import torch
from torch import nn

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchmetrics import HingeLoss

from models import FashionMLP, FashionCNN, FashionSVM, LeastSquaresClassifier
from dataset import IN_FEATURES, NUM_CLASSES, get_dataset
from utils import train, valid_epoch

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main(l_rate, gamma, stop_layer, n_epochs, k, batch_size,save_path, load_path, seed):
    # replicability
    torch.manual_seed(seed)

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_set, test_set = get_dataset()

    splits=KFold(n_splits=k, shuffle=True,random_state=seed)
    best_accuracy = -1

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_set)))):
        print('Fold {}'.format(fold + 1))

        # load train & validation loader
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler)
        
        # load model
        model = FashionCNN()
        model.to(device)

        # train
        #history = train(model, train_loader, valid_loader, loss_fn, device,  lr=l_rate, n_epochs=n_epochs, gamma=gamma)
        #torch.save(model.state_dict(), save_path)
        model.load_state_dict(torch.load(load_path))

        # save layer2 output
        getattr(model,stop_layer).register_forward_hook(get_activation(stop_layer))

        output_train_layer2 = []
        train_labels = []
        model.eval()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            train_labels.append(labels)
            output_train_layer2.append(activation[stop_layer].flatten(1))
        
        # concatenate output and labels
        output_train_layer2 = torch.cat(output_train_layer2)
        train_labels = torch.cat(train_labels)

        # train svm classifier
        rf = LeastSquaresClassifier()
        rf.fit(output_train_layer2, train_labels)

        # get validation accuracy
        val_accuracy = 0.0
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            val_accuracy += rf.score(activation[stop_layer].flatten(1), labels)

        val_accuracy/=len(valid_loader)

        if best_accuracy < val_accuracy:
            pickle.dump(rf, open(save_path, 'wb'))

    # load best model
    model = FashionCNN()
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    getattr(model,stop_layer).register_forward_hook(get_activation(stop_layer))
    rf = pickle.load(open(save_path, 'rb'))

    # test
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_accuracy = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        val_accuracy += rf.score(activation[stop_layer].flatten(1), labels)

    test_accuracy/=len(test_loader)
    print(f"Test loss: {test_loss}; Test accuracy: {test_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train2.py')
    parser.add_argument('--l_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default='0.9', help='gamma parameter for optimizer scheduler')   
    parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--k', type=int, default=4, help='k parameter in k-fold validation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--load_path', type=str, default='models/best_model_cnn.pth', help='best model saved path')
    parser.add_argument('--save_path', type=str, default='models/best_model_svm.sav')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--stop_layer', type=str, default='fc2')
    kwargs = parser.parse_args()

    main(**vars(kwargs))
