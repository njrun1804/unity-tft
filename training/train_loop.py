import torch, torch.nn.functional as F

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_sum, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        opt.zero_grad()
        pred = model(X).squeeze()
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(X)
        n += len(X)
    return loss_sum / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        pred = model(X).squeeze()
        loss_sum += F.mse_loss(pred, y, reduction="sum").item()
        n += len(X)
    return loss_sum / n
