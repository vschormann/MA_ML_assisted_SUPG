import time
from example_2_1_dataloader import test_dataset, test_loader
from example_2_1_model import model
import torch


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



start = time.perf_counter()
input = test_dataset[0]['x']
nn=model(dtype=input.dtype, device=input.device, dir='data/example_2_1/models/nn_init.pth')
optimizer = torch.optim.Adam(nn.parameters())

for i in range(1):
    train_loop(dataloader=test_loader, nn=nn, optimizer=optimizer)
t1 = 20*(time.perf_counter() - start)
disp1 = f"example 2.1 model training on test set took {t1:.3f} seconds"
print(disp1)



from IPython.display import clear_output
start = time.perf_counter()



dataset = test_dataset
iterations = 20

size = len(dataset)
train_loss = 0
for i in range(size):
    fs = dataset[i]['fem_solver']
    params = dataset[i]['params']
    optimizer = torch.optim.Adam([params])
    train_loss = 0
    clear_output(wait=True)
    for steps in range(iterations):
        optimizer.zero_grad()
        loss = fs(params)
        loss.backward()
        # Backpropagation
        optimizer.step()

    train_loss += loss
    target = params
    #torch.save(target, f'data/example_2_1/test_set/target_values/t_{i}.pt')
    clear_output(wait=True)
    print(f"Current datapoint: {i} \n")
    print(f"loss: {loss}")
train_loss /= size
print(f"train_loss: {train_loss:>7f}")
t2 = time.perf_counter() - start
disp2 = f"Section 2.3.1: computing target values with 500 Adam steps took {t2:.3f} seconds"
print(disp2)