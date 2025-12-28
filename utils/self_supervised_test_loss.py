import numpy as np
import torch

def test_loop(dataloader, nn):
    nn.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, fem_solver in dataloader:
            batch_size = len(X)
            for idx in range(len(X)):
            # Compute prediction and loss
                z2 = nn(X[idx])
                test_loss += 1/batch_size*fem_solver[idx](z2).item()

    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")
    return(test_loss)


def test(dataloader, model, model_nm, epochs, dtype, device, test_loss_file):
    test_loss_arrray = np.array([])
    for idx in range(epochs):
        nn = model(device=device, dtype=dtype, dir=f"{model_nm}_{idx}.pth")
        print(f"Epoch {idx+1}\n-------------------------------")
        test_loss_arrray = np.append(test_loss_arrray,test_loop(dataloader,  nn))

    np.save(file=test_loss_file, arr=test_loss_arrray)