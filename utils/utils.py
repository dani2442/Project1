import torch
from tqdm import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        predictions = torch.argmax(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss/len(dataloader.sampler), train_correct/len(dataloader.sampler)


def valid_epoch(model,dataloader,loss_fn, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        predictions = torch.argmax(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss/len(dataloader.sampler), val_correct/len(dataloader.sampler)


def train(model, train_loader, valid_loader, loss_fn, device, save_path, lr=1e-4, n_epochs=5, gamma=0.9):
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    best_accuracy = -1.0
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss, train_acc=train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc=valid_epoch(model, valid_loader, loss_fn, device)
        
        scheduler.step()
        pbar.set_description(f"Training Loss: {train_loss:.3f} Valid Loss: {valid_loss:.3f} Training Acc: {train_acc:.2f} Valid Acc: {valid_acc:.2f}")

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        if best_accuracy < valid_acc:
            best_accuracy = valid_acc
            torch.save(model.state_dict(), save_path)

    return history


def train_final_layer(model, preprocessing, train_loader, valid_loader, loss_fn, device, lr=1e-4, n_epochs=5, gamma=0.9):
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss, train_acc=train_epoch(model, train_loader, loss_fn,optimizer, device)
        valid_loss, valid_acc=valid_epoch(model, valid_loader, loss_fn, device)
        
        scheduler.step()
        pbar.set_description(f"Training Loss: {train_loss:.3f} Valid Loss: {valid_loss:.3f} Training Acc: {train_acc:.2f} Valid Acc: {valid_acc:.2f}")

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

    return history