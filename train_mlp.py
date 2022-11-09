import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.models import FashionMLP, FashionCNN
from utils.dataset import IN_FEATURES, NUM_CLASSES, get_dataset
from utils.utils import train, valid_epoch, calculate_confusion_matrix, plot_confusion_matrix


def main(l_rate, gamma, n_epochs, k, batch_size, save_path, seed):
    # replicability
    torch.manual_seed(seed)

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # load dataset
    train_set, test_set = get_dataset()

    splits=KFold(n_splits=k, shuffle=True,random_state=seed)
    foldperf={}

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_set)))):
        print('Fold {}'.format(fold + 1))

        # load train & validation loader
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler)
        
        # select model
        model = FashionCNN()
        model.to(device)

        # train
        history = train(model, train_loader, valid_loader, loss_fn, device, save_path, lr=l_rate, n_epochs=n_epochs, gamma=gamma)

        foldperf['fold{}'.format(fold+1)] = history  

    # load best model
    model = FashionCNN()
    model.to(device)
    model.load_state_dict(torch.load(save_path))

    # test
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loss, test_accuracy = valid_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss}; Test accuracy: {test_accuracy}")
    print(f"Training Summary: {history}")

    # get confusion matrix
    m = calculate_confusion_matrix(model, test_loader, device)
    plot_confusion_matrix(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--l_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma parameter for optimizer scheduler')   
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--k', type=int, default=4, help='k parameter in k-fold validation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--save_path', type=str, default='models/best_model_cnn.pth', help='best model saved path')
    parser.add_argument('--seed', type=int, default=1234)
    kwargs = parser.parse_args()

    main(**vars(kwargs))
