import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
import xgboost as xgb

# Load and clean data
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

# Select target and features
swap = ['max_pore_size_diameter','max_pore_size_volume']
target = 'homo_lumo_gap_eV'
exclude = ['filename', 'inner_atom', target]
if target in swap:
    exclude += swap
features = [col for col in df.columns if col not in exclude]

X = df[features]
y = df[target]
mop_ids = df['filename']

# Split into train/test
X_train, X_test, y_train, y_test, mop_train, mop_test = train_test_split(
    X, y, mop_ids, test_size=0.2
)

# Ensemble of XGBoost models
n_models = 20
params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.7,
    'objective': 'reg:squarederror',
}

preds_list = []

for i in range(n_models):
    X_res, y_res = resample(X_train, y_train)
    model = xgb.XGBRegressor(**params)
    model.fit(X_res, y_res)
    preds = model.predict(X_test)
    preds_list.append(preds)

# Stack predictions: [n_models, n_samples]
preds_arr = np.array(preds_list)
ensemble_mean = preds_arr.mean(axis=0)
ensemble_std = preds_arr.std(axis=0)

# Evaluate
true_vals = y_test.values
abs_error = np.abs(true_vals - ensemble_mean)
mse = mean_squared_error(true_vals, ensemble_mean)
r2 = r2_score(true_vals, ensemble_mean)
mae = mean_absolute_error(true_vals, ensemble_mean)

print(f"\nTest Set Metrics (XGBoost Ensemble):\nR² = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}")

# Coverage at specified number of std deviations
n_deviations_for_coverage = 3
lower_bound = ensemble_mean - ensemble_std * n_deviations_for_coverage
upper_bound = ensemble_mean + ensemble_std * n_deviations_for_coverage
coverage = ((true_vals >= lower_bound) & (true_vals <= upper_bound)).mean()
print(f"±{n_deviations_for_coverage}σ Ensemble coverage: {coverage*100:.2f}%")

# Plot parity with error bars
plt.figure(figsize=(8, 6))
# Modify yerr to use n_deviations_for_coverage
plt.errorbar(true_vals, ensemble_mean, yerr=ensemble_std * n_deviations_for_coverage, fmt='o', alpha=0.6, capsize=3)
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Ensemble Mean Prediction")
plt.title(f"XGBoost Ensemble Parity Plot\nCoverage (±{n_deviations_for_coverage}σ): {coverage*100:.2f}%, R² = {r2:.2f}")
plt.tight_layout()
plt.show()

# Error summary
error_df = pd.DataFrame({
    'true': true_vals,
    'pred': ensemble_mean,
    'std': ensemble_std,
    'abs_error': abs_error,
    f'covered_pm{n_deviations_for_coverage}std': (true_vals >= lower_bound) & (true_vals <= upper_bound),
    'mop_id': mop_test.values
})

print("\nWorst predictions by absolute error:")
print(error_df.sort_values(by='abs_error', ascending=False).head(10).to_string(index=False, float_format="%.3f"))