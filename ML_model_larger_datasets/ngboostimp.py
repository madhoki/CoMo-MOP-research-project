import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.utils import resample
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

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

swap = ['max_pore_size_diameter','max_pore_size_volume']
# target = 'max_pore_size_diameter'
target = 'homo_lumo_gap_eV'

exclude = ['filename', 'inner_atom', target]
if target in swap:
    exclude += swap
features = [col for col in df.columns if col not in exclude]

X = df[features]
y = df[target]
mop_ids = df['filename']

X_train, X_test, y_train, y_test, mop_train, mop_test = train_test_split(
    X, y, mop_ids, test_size=0.2, random_state=42
)

n_models = 10
mean_preds_list = []
std_preds_list = []

for i in range(n_models):
    print(f'Num sampled: {i+1}')
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    ngb = NGBRegressor(Dist=Normal, verbose=False, random_state=i)
    ngb.fit(X_resampled, y_resampled)
    preds = ngb.pred_dist(X_test)
    mean_preds_list.append(preds.loc)
    std_preds_list.append(preds.scale)

mean_preds_arr = np.array(mean_preds_list)
std_preds_arr = np.array(std_preds_list)

ensemble_mean = mean_preds_arr.mean(axis=0)


ensemble_var = (std_preds_arr**2).mean(axis=0) + mean_preds_arr.var(axis=0)
ensemble_std = np.sqrt(ensemble_var)

true_vals = y_test.values
abs_error = np.abs(true_vals - ensemble_mean)
mse = mean_squared_error(true_vals, ensemble_mean)
r2 = r2_score(true_vals, ensemble_mean)
mae = mean_absolute_error(true_vals, ensemble_mean)
corr, _ = pearsonr(ensemble_std, abs_error)

print(f"\nTest Set Metrics (NGBoost Ensemble):\nR² = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")
print(f"📊 Correlation between predicted std and actual error: r = {corr:.3f}")


reg = Ridge()
reg.fit(ensemble_std.reshape(-1, 1), abs_error)
calibrated_std = reg.predict(ensemble_std.reshape(-1, 1))
calibrated_std = np.clip(calibrated_std, 1e-4, None)

lower = ensemble_mean - calibrated_std
upper = ensemble_mean + calibrated_std
coverage = ((true_vals >= lower) & (true_vals <= upper)).mean()
print(f"✅ Calibrated ±1σ coverage = {coverage*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.errorbar(true_vals, ensemble_mean, yerr=calibrated_std, fmt='o', alpha=0.6, capsize=3)
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Ensemble Predicted Means")
plt.title(f"NGBoost Ensemble Parity Plot with Calibrated Error Bars\nCoverage: {coverage*100:.2f}%, r(std, abs_error) = {corr:.2f}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(ensemble_std, abs_error, alpha=0.5)
plt.plot(ensemble_std, reg.predict(ensemble_std.reshape(-1, 1)), color='red', label='Calibrated mapping')
plt.xlabel("Ensemble Predicted Std Dev")
plt.ylabel("Absolute Error")
plt.title("Ensemble Uncertainty Calibration")
plt.legend()
plt.tight_layout()
plt.show()

error_df = pd.DataFrame({
    'true': true_vals,
    'pred': ensemble_mean,
    'std_raw': ensemble_std,
    'std_calibrated': calibrated_std,
    'abs_error': abs_error,
    'covered': (true_vals >= lower) & (true_vals <= upper),
    'mop_id': mop_test.values
})

print("\nWorst predictions by absolute error:")
print(error_df.sort_values(by='abs_error', ascending=False).head(10).to_string(index=False, float_format="%.3f"))
