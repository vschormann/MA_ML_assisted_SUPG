import torch
    

class md1(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        if isinstance(weights, torch.Tensor):
            self.weights = torch.nn.Parameter(weights)
        else:
            self.weights = torch.randn(weights)
    
    def forward(self):
        return self.weights
    

class md2(torch.nn.Module):
    def __init__(self, chnl_num, cell_num):
        super().__init__()
        self.pooling = torch.nn.MaxPool1d(cell_num)
        self.flatten = torch.nn.Flatten()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(chnl_num, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, cell_num)
        )

    def forward(self, input):
        reduced = self.pooling(input)
        x = self.flatten(reduced)
        return self.net(x)