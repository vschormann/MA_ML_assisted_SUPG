from section_2_3_2_supervised_dataset import train_dataset, train_loader, test_loader, test_dataset
from example_2_1_dataloader import test_loader

from example_2_1_model import model

import numpy as np
import torch
import torch

from utils.self_supervised_test_loss import test_loop



def train_loop(dataloader, nn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nn.train()
    train_loss = 0
    loss_fn = torch.nn.MSELoss()
    for batch, (X, T) in enumerate(dataloader):
        agg_loss = 0
        batch_size = len(X)
        for idx in range(batch_size):
        # Compute prediction and loss
            z2 = nn(X[idx])
            t = T[idx]
            loss = loss_fn(z2, t)/batch_size
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




def train(model_dir, model_init, optim, dtype, device, epochs):
    nn = model(device=device, dtype=dtype, dir=model_init)
    optimizer = optim(nn.parameters())
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_arrray = np.append(train_loss_arrray, train_loop(train_loader, nn, optimizer))
        torch.save(nn.state_dict(), f'{model_dir}_{t}.pth')


def test(model_nm, epochs, dtype, device, test_loss_file):
    test_loss_arrray = np.array([])
    for idx in range(epochs):
        nn = model(device=device, dtype=dtype, dir=f"{model_nm}_{idx}.pth")
        print(f"Epoch {idx+1}\n-------------------------------")
        test_loss_arrray = np.append(test_loss_arrray,test_loop(test_loader,  nn))

    np.save(file=test_loss_file, arr=test_loss_arrray)

input = train_dataset[0]['x']


train(
         model_dir='data/section_2_3_2/models/nn_supervised_Adam',
         model_init='data/example_2_1/models/nn_init.pth',
         optim=torch.optim.Adam,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         )
#compute the train loss
test(
        model_nm='data/section_2_3_2/models/nn_supervised_Adam',
        epochs=20,
        dtype=input.dtype,
        device=input.device,
        test_loss_file='data/section_2_3_2/supervised_train_loss_Adam.npy'
        )
#compute the test loss
test(
        model_nm='data/section_2_3_2/models/nn_supervised_Adam',
        epochs=20,
        dtype=input.dtype,
        device=input.device,
        test_loss_file='data/section_2_3_2/supervised_test_loss_Adam.npy'
        )

print("Done!")