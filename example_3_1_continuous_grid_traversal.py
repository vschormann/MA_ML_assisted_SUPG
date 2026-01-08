import torch
from torch import nn

from utils.supervised_dataset import supervised_dataset as dataset
from torch.utils.data import DataLoader
from utils.transforms import flattening_transform, reindex_transform

def create_cell_ind_to_grid(H, W):
    w, h = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )

    d = (H - 1 - w) + h          # diagonal index
    order = torch.argsort(
        d.flatten() * max(H, W) + w.flatten()
    )

    out = torch.empty(H * W, dtype=torch.long)
    out[order] = torch.arange(H * W)

    return out.view(H, W)

def create_flattening_index_set(H,W, continuous_traversal=False):
    h, w = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )
    if continuous_traversal:
        diag_mask = h+w
        diag_mask = torch.ones_like(diag_mask)-diag_mask%2

        sgn_fn = torch.where(diag_mask%2==1, 1, -1)

        diag_key = ((1+sgn_fn/(2*H))*h + w).flip(0)
    else:
        diag_key = (h + (1+1/(2*W))*(w)).flip(0)

    return torch.argsort(diag_key.flatten())



def train_loop(dataloader, nn, optimizer, loss_fn):
    num_batches = len(dataloader)
    nn.train()
    train_loss = 0.0
    for batch, (x, t) in enumerate(dataloader):
        y_pred = nn(x)

        loss = loss_fn(y_pred,t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= num_batches
    print(f"Train loss: {train_loss:>7f}")
    return(train_loss)

def test_loop(dataloader, nn, loss_fn):
    nn.eval()
    num_batches = len(dataloader)
    test_loss = 0.0

    with torch.no_grad():
        for x, t in dataloader:
            y_pred = nn(x)

            loss = loss_fn(y_pred,t)

            test_loss += loss.item()

    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")
    return(test_loss)


class resblock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        z = self.layer(x)
        return z+x

class MLP_block(nn.Module):
    def __init__(self, insize, outsize, activation, normalization=None, residual_connection=None):
        super().__init__()
        self.residual_connection = residual_connection
        if normalization:
            self.block = nn.Sequential(
                nn.Linear(insize, outsize),
                activation(),
                normalization(outsize)
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(insize, outsize),
                activation(),
            )

            
    def forward(self, x):
        z = self.block(x) 
        if self.residual_connection:
            z += x
        return z
    
class const_featuresize_MLP(nn.Module):
    def __init__(self, depth, features, activation,residual_connection=False, normalization=False):
        super().__init__()
        res_layers = []
        for _ in range(depth):
            res_layers.append(MLP_block(features, features, activation=activation, residual_connection=residual_connection, normalization=normalization))
        self.res_net = nn.Sequential(*res_layers)

    def forward(self, x):
        return self.res_net(x)

    

device = 'mps'
dtype = torch.float32
depth = 3
dir = f"data/example_3_1/models/d{depth}_model_init.pth"
residual_connection = False
normalization = None
in_features = 15028
out_features = 1024
activation = nn.LeakyReLU

class sMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, activation, cdim_MLP_depth, residual_connection=False, normalization=False, dir=None):
        super().__init__()
        self.ffwd = nn.Sequential(
            MLP_block(in_features, out_features, activation=activation),
            const_featuresize_MLP(depth=cdim_MLP_depth, features=out_features, activation=activation, residual_connection=residual_connection, normalization=normalization)
        )
        if dir:
            self.load_state_dict(torch.load(dir, weights_only=True))

    def forward(self, x):
        y_pred = self.ffwd(x)
        return y_pred


H,W = 32,32
transform_flat_key = create_flattening_index_set(H=H+2,W=W+2, continuous_traversal=True)

transform = flattening_transform(flat_key=transform_flat_key, C=13)

cell_ind_to_grid = create_cell_ind_to_grid(H,W)
flat_key = create_flattening_index_set(H,W, True)

target_reindex = cell_ind_to_grid.flatten()[flat_key]
target_transform = reindex_transform(reindex=target_reindex)


train_set = dataset(
    input_dir="data/example_3_1/training_set/inputs/", 
    target_dir="data/example_3_1/training_set/target_values/", 
    transform=transform, 
    target_transform=target_transform, 
    device=device
)

test_set = dataset(
    input_dir="data/example_3_1/test_set/inputs/", 
    target_dir="data/example_3_1/test_set/target_values/", 
    transform=transform, 
    target_transform=target_transform, 
    device=device
)

model = sMLP(
    in_features=in_features, 
    out_features=out_features, 
    activation=activation, 
    cdim_MLP_depth=depth, 
    residual_connection=residual_connection, 
    normalization=normalization,
    dir=dir
).to(device=device, dtype=dtype)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters())
step_size = 10
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size)

batch_size = 10
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

for i in range(30):
    print(f'epoch: {i}')
    train_loop(dataloader=train_loader,nn=model,optimizer=optimizer,loss_fn=loss_fn)
    test_loop(dataloader=test_loader, nn=model, loss_fn=loss_fn)
    scheduler.step()