"""Small training loops with explicit models, losses, and datasets."""

import torch


def train_epoch(model, loader, optimizer, device, loss_fn=None):
    """Train for one supervised epoch and return mean batch loss."""

    model.train()
    loss_fn = loss_fn or torch.nn.functional.mse_loss
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch")
    return total / len(loader)


train = train_epoch


def self_supervised_train(model, loader, optimizer, loss_fn, device):
    """Train for one epoch through a batched differentiable FEM loss."""

    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(data.ptr, data.mesh_id, output)
        loss.backward()
        optimizer.step()
        total += loss.item()
    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch")
    return total / len(loader)


def fit(model, loader, optimizer, epochs, device, loss_fn=None, physics=False):
    """Run multiple epochs and return a plain list of mean losses."""

    epoch = self_supervised_train if physics else train_epoch
    history = []
    for _ in range(epochs):
        if physics:
            history.append(epoch(model, loader, optimizer, loss_fn, device))
        else:
            history.append(epoch(model, loader, optimizer, device, loss_fn=loss_fn))
    return history
