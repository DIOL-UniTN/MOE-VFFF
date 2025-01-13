import torch

from tqdm import tqdm

def train_epoch(model, optim, trainloader, criterion, device):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for images, targets in tqdm(trainloader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        total += targets.size(0)
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
    accuracy = correct/total
    return running_loss/len(trainloader), accuracy

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, targets).item()
        total += targets.size(0)
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
    accuracy = correct/total
    return running_loss/len(loader), accuracy

def stop_earlier(patience, patience_counter, val_loss, val_acc, best_loss, best_acc):
    early_stopped = False
    patience_counter += 1
    if val_loss < best_loss :
        patience_counter, best_loss = 0, val_loss
    if val_acc > best_acc:
        patience_counter, best_acc = 0, val_acc
    if patience_counter >= patience:
        early_stopped = True
    return patience_counter, best_loss, best_acc, early_stopped
