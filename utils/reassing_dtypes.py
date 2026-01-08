import os
import torch


def reassign_dtypes(set, dtype):
    for idx in range(len(set)):
        input_path = os.path.join(set.input_dir, f"x_{idx}.pt")
        input = torch.load(input_path) 

        target_path = os.path.join(set.target_dir, f"t_{idx}.pt")
        target = torch.load(target_path) 

        x = input.to(torch.float32)
        torch.save(x, input_path)
        t = target.to(torch.float32)
        torch.save(t, target_path)
