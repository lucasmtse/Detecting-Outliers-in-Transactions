import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.Linear(4, encoding_dim),
            nn.ELU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(encoding_dim,4),
            nn.ELU(),
            nn.Linear(4,8),
            nn.ELU(),
            nn.Linear(8,16),
            nn.ELU(),
            nn.Linear(16,input_dim),
            nn.ELU(),
        )

    def forward(self, x):
        # Sure x is of shape (batch_size, input_dim)
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}")
        x = self.encoder(x)  
        x = self.decoder(x)
        return x

import numpy as np
import torch
from torch import nn
import tqdm

def train_step_fn(model, loss_fn, optimizer, device):
    def step(x):
        model.train()
        x = x.to(device)
        x_hat = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return step

def val_step_fn(model, loss_fn, device):
    def step(x):
        model.eval()
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
        return loss.item()
    return step

def fit(model, loss_fn, optimizer, x_train, x_val, epochs=100, batch_size=256, patience=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_step = train_step_fn(model, loss_fn, optimizer, device)
    val_step = val_step_fn(model, loss_fn, device)

    train_losses, val_losses = [], []
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in (range(epochs)):
        # shuffle
        perm = torch.randperm(x_train.size(0))
        x_train = x_train[perm]

        # mini-batches
        bt_losses = []
        for i in tqdm.tqdm(range(0, x_train.size(0), batch_size)):
            bt_losses.append(train_step(x_train[i:i+batch_size]))
        avg_train = float(np.mean(bt_losses))
        train_losses.append(avg_train)

        # val
        bv_losses = []
        for i in range(0, x_val.size(0), batch_size):
            bv_losses.append(val_step(x_val[i:i+batch_size]))
        avg_val = float(np.mean(bv_losses))
        val_losses.append(avg_val)

        # best / early stopping
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch+1) % 1 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train {avg_train:.6f} | Val {avg_val:.6f} | Best {best_val:.6f}")

        if patience is not None and no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #return train_losses, val_losses

