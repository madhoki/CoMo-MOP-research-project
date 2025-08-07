import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('ML_model_larger_datasets/data/enriched_mop_data.csv')
df = df.replace('error', pd.NA).dropna()

for col in [
    'avg_metal_dist', 'inner_diameter', 'inner_volume',
    'max_pore_size_diameter', 'max_pore_size_volume',
    'homo_lumo_gap_eV', 'mol_wt', 'num_rings', 'num_aromatic_rings',
    'num_rotatable_bonds', 'num_h_donors', 'num_h_acceptors',
    'tpsa', 'logp', 'heavy_atom_count', 'fraction_sp3_carbons','asphericity'
]:
    df[col] = df[col].astype(float)

target = 'max_pore_size_diameter' 
target = 'homo_lumo_gap_eV' 
def paired_feature(target):
    if 'diameter' in target:
        return target.replace('diameter', 'volume')
    elif 'volume' in target:
        return target.replace('volume', 'diameter')
    return None
exclude = ['filename', 'inner_atom', target]
paired = paired_feature(target)
if paired in df.columns:
    exclude.append(paired)
features = [col for col in df.columns if col not in exclude]

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

def make_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

model = make_model(X.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.tensor(X_test))
        val_loss = loss_fn(val_pred, torch.tensor(y_test)).item()
        val_losses.append(val_loss)
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

model.eval()
y_pred = model(torch.tensor(X_test)).detach().numpy().flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.3f}, R2: {r2:.3f}')

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'Parity Plot\n$R^2$={r2:.3f}, MSE={mse:.3f}')
plt.tight_layout()
plt.show()

train_sizes = [0.5, 0.6, 0.7, 0.8, 0.9]
r2_scores = []
mse_scores = []
for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    model = make_model(X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    y_pred = model(torch.tensor(X_test)).detach().numpy().flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)

plt.figure()
plt.plot([int(s*100) for s in train_sizes], r2_scores, marker='o', label='R2')
plt.plot([int(s*100) for s in train_sizes], mse_scores, marker='s', label='MSE')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Score')
plt.title('Training Set Size Optimisation')
plt.legend()
plt.tight_layout()
plt.show()
