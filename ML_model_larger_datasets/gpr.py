import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('ML_model_larger_datasets/data/enriched_mop_data.csv')
df = df.replace('error', pd.NA).dropna()

for col in [
    'avg_metal_dist', 'inner_diameter', 'inner_volume',
    'max_pore_size_diameter', 'max_pore_size_volume',
    'homo_lumo_gap_eV', 'mol_wt', 'num_rings', 'num_aromatic_rings',
    'num_rotatable_bonds', 'num_h_donors', 'num_h_acceptors',
    'tpsa', 'logp', 'heavy_atom_count', 'fraction_sp3_carbons', 'asphericity'
]:
    df[col] = df[col].astype(float)

target = 'homo_lumo_gap_eV'
exclude = ['filename', 'inner_atom', target]
features = [col for col in df.columns if col not in exclude]

X = df[features]
y = df[target]
mop_ids = df['filename']

X_train_full, X_test, y_train_full, y_test, mop_train, mop_test = train_test_split(
    X, y, mop_ids, test_size=0.2, random_state=42
)

max_train = 500
if len(X_train_full) > max_train:
    X_train = X_train_full.sample(n=max_train, random_state=42)
    y_train = y_train_full.loc[X_train.index]
else:
    X_train, y_train = X_train_full, y_train_full

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca_components = min(X_train_scaled.shape[1], 10)  
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)

gpr.fit(X_train_pca, y_train)

y_pred, y_std = gpr.predict(X_test_pca, return_std=True)
true_vals = y_test.values
abs_error = np.abs(true_vals - y_pred)

mse = mean_squared_error(true_vals, y_pred)
r2 = r2_score(true_vals, y_pred)
mae = mean_absolute_error(true_vals, y_pred)
coverage = ((true_vals >= (y_pred - y_std)) & (true_vals <= (y_pred + y_std))).mean()

print(f"\nGPR with PCA and Scaling:\nR² = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")
print(f"✅ ±1σ predictive coverage: {coverage*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.errorbar(true_vals, y_pred, yerr=y_std, fmt='o', alpha=0.6, capsize=3)
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Means")
plt.title(f"GPR Parity Plot with Calibrated Uncertainty\nCoverage: {coverage*100:.2f}%, R² = {r2:.2f}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_std, abs_error, alpha=0.5)
plt.xlabel("Predicted Std Dev")
plt.ylabel("Absolute Error")
plt.title("GPR: Std Dev vs Absolute Error")
plt.tight_layout()
plt.show()

error_df = pd.DataFrame({
    'true': true_vals,
    'pred': y_pred,
    'std': y_std,
    'abs_error': abs_error,
    'covered': (true_vals >= (y_pred - y_std)) & (true_vals <= (y_pred + y_std)),
    'mop_id': mop_test.values
})

print("\nWorst predictions by absolute error:")
print(error_df.sort_values(by='abs_error', ascending=False).head(10).to_string(index=False, float_format="%.3f"))
