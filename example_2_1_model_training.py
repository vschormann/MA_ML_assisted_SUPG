from example_2_1_dataloader import train_dataset, train_loader, test_loader
from example_2_1_model import model

import numpy as np
import torch
import time

from utils.self_supervised_test_loss import test



def train_loop(dataloader, nn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nn.train()
    train_loss = 0
    for batch, (X, fem_solver) in enumerate(dataloader):
        agg_loss = 0
        batch_size = len(X)
        for idx in range(batch_size):
        # Compute prediction and loss
            z2 = nn(X[idx])
            loss = 1/batch_size*fem_solver[idx](z2)
            loss.backward()
            agg_loss += loss.item()
        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = agg_loss, batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss += agg_loss
    train_loss /= num_batches
    print(f"train_loss: {train_loss:>7f}")
    return(train_loss)



def train(model_dir, model_init, optim, dtype, device, epochs, train_loss_file):
    nn = model(device=device, dtype=dtype, dir=model_init)
    optimizer = optim(nn.parameters())
    train_loss_arrray = np.array([])
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_arrray = np.append(train_loss_arrray, train_loop(train_loader, nn, optimizer))
        torch.save(nn.state_dict(), f'{model_dir}_{t}.pth')

    np.save(file=train_loss_file, arr=train_loss_arrray)




input = train_dataset[0]['x']
start = time.perf_counter()
train(
         model_dir='data/example_2_1/models/nn_exact_SGD',
         model_init='data/example_2_1/models/nn_init.pth',
         optim=torch.optim.SGD,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         train_loss_file='data/example_2_1/exact_train_loss_SGD.npy'
         )

t1 = time.perf_counter() - start
print(f"Example 2.1: 20 epochs with SGD took {t1:.3f} seconds")

start = time.perf_counter()
train(
         model_dir='data/example_2_1/models/nn_exact_Adam',
         model_init='data/example_2_1/models/nn_init.pth',
         optim=torch.optim.Adam,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         train_loss_file='data/example_2_1/exact_train_loss_Adam.npy'
         )

t1 = time.perf_counter() - start
print(f"Example 2.1: 20 epochs with SGD took {t1:.3f} seconds")

test(   
        dataloader=test_loader,
        model=model,
        model_nm='data/example_2_1/models/nn_exact_Adam',
        epochs=20,
        dtype=input.dtype,
        device=input.device,
        test_loss_file='data/example_2_1/exact_test_loss_Adam.npy'
        )

test(
        dataloader=test_loader,
        model=model,
        model_nm='data/example_2_1/models/nn_exact_SGD',
        epochs=20,
        dtype=input.dtype,
        device=input.device,
        test_loss_file='data/example_2_1/exact_test_loss_SGD.npy'
        )
print("Done!")



