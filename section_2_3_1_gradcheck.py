from example_2_1_dataloader import test_dataset
from example_2_1_model import model

import torch
import numpy as np
import matplotlib.pyplot as plt



input = test_dataset[0]['x']
eps = 1e-4
nn = model(device=input.device, dtype=input.dtype, dir='data/example_2_1/models/nn_init.pth')

grad_err = []
up = []
low = []
for j in range(10):
    func = test_dataset[j]['fem_solver']
    input = test_dataset[j]['x']
    out = nn(input)
    x = out.clone().detach()
    x.requires_grad = True
    num_grad = torch.zeros(len(x))
    for i in range(x.numel()):
        x_pos = x.clone().detach()
        x_neg = x.clone().detach()

        x_pos[i] += eps
        x_neg[i] -= eps

        num_grad[i] = (func(x_pos) - func(x_neg)) / (2 * eps)


    x.grad = None
    func(x).backward()

    max_len = 0
    for i in range(input.shape[1]):
        cell = input[:6,i]
        a = torch.sqrt(cell[0]**2 + cell[1]**2)
        if a > max_len:
            max_len = a.item()
        b= torch.sqrt(cell[2]**2 + cell[3]**2)
        if b > max_len:
            max_len = b.item()
        c = torch.sqrt(cell[4]**2 + cell[5]**2)
        if c > max_len:
            max_len = c.item()
    min_len = 1
    for i in range(input.shape[1]):
        cell = input[:6,i]
        cell_diam = 0
        a = torch.sqrt(cell[0]**2 + cell[1]**2)
        if a > cell_diam:
            cell_diam = a.item()
        b= torch.sqrt(cell[2]**2 + cell[3]**2)
        if b > cell_diam:
            cell_diam = b.item()
        c = torch.sqrt(cell[4]**2 + cell[5]**2)
        if c > cell_diam:
            cell_diam = c.item()
        if cell_diam < min_len:
            min_len = cell_diam
    grad_err.append(torch.max(torch.abs(x.grad - num_grad)).item())
    low.append(min_len)
    up.append(max_len)

low = np.array(low)
up = np.array(up)
grad_err = np.array(grad_err)
arr = np.vstack((low, up, grad_err))

grad_loss_file = 'data/example_2_1/grad_loss_1e4.npy'
np.save(file=grad_loss_file, arr=arr)


grad_diff = 'data/example_2_1/grad_loss_1e4.npy'


grad_diff = np.load(grad_diff)



plt.scatter(x=grad_diff[1], y=grad_diff[0], sizes=3*1e+8*grad_diff[2], c=grad_diff[2], cmap='viridis')


plt.colorbar()
plt.grid(True)
plt.legend()
plt.savefig('section_2_3_1_grad_diff.png')
plt.show()