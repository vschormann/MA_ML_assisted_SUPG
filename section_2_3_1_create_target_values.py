from IPython.display import clear_output
import torch
from section_2_3_1_dataset import train_dataset

dataset = train_dataset
iterations = 500

size = len(dataset)
train_loss = 0
for i in range(size):
    fs = train_dataset[i]['fem_solver']
    params = train_dataset[i]['params']
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
    torch.save(target, f'data/example_2_1/training_set/target_values/t_{i}.pt')
    clear_output(wait=True)
    print(f"Current datapoint: {i} \n")
    print(f"loss: {loss}")
train_loss /= size
print(f"train_loss: {train_loss:>7f}")