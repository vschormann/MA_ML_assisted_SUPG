from example_2_1_dataloader import train_dataset, train_loader, test_loader
from example_2_1_model import model

import numpy as np
import torch



def train_loop(dataloader, nn, lr, iterations):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nn.train()
    train_loss = 0
    for batch, (FEniCSx_solver) in enumerate(dataloader):
        agg_loss = 0
        batch_size = len(fs)
        for idx in range(batch_size):
            for steps in range(iterations):
            # Compute prediction and loss
                fs = FEniCSx_solver[idx]
                loss = fs.loss()
                weights = fs.yh.y.array
                weights -= lr*fs.grad()
                fs.set_weights(weights)

        if batch % 10 == 0:
            loss, current = agg_loss, batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss += agg_loss
    train_loss /= num_batches
    print(f"train_loss: {train_loss:>7f}")
    return(train_loss)


def test_loop(dataloader, nn):
    nn.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for FEniCSx_solver in dataloader:
            batch_size = len(FEniCSx_solver)
            for idx in range(batch_size):
            # Compute prediction and loss
                fs = FEniCSx_solver[idx]
                test_loss += 1/batch_size*fs.loss()

    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")
    return(test_loss)


def train(model, model_dir, model_init, optimizer, dtype, device, epochs, train_loss_file, test_loss_file):
    nn = model(device=device, dtype=dtype, dir=model_init)
    optimizer = optimizer(nn.paramters())
    train_loss_arrray = np.array([])
    test_loss_arrray = np.array([])
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_arrray = np.append(train_loss_arrray, train_loop(train_loader, nn, optimizer))
        test_loss_arrray = np.append(test_loss_arrray,test_loop(test_loader,  nn))
        torch.save(nn.state_dict(), f'{model_dir}_{t}.pth')

    np.save(file=train_loss_file, arr=train_loss_arrray)

    np.save(file=test_loss_file, arr=test_loss_arrray)


input = train_dataset[0]['x']

train(model=model,
         model_dir='data/example_2_1/models/nn_exact_SGD',
         model_init='data/example_2_1/models/nn_init.pth',
         optimizer=torch.optim.SGD,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         train_loss_file='data/example_2_1/exact_train_loss_SGD.npy',
         test_loss_file='data/example_2_1/exact_testloss_SGD.npy'
         )

train(model=model,
         model_dir='data/example_2_1/models/nn_exact_Adam',
         model_init='data/example_2_1/models/nn_init.pth',
         optimizer=torch.optim.Adam,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         train_loss_file='data/example_2_1/exact_train_loss_Adam.npy',
         test_loss_file='data/example_2_1/exact_testloss_Adam.npy'
         )


print("Done!")



