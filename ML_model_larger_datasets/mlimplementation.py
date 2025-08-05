import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load data
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

# Choose target and features
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

X = df[features]
y = df[target]
mop_ids = df['filename']  # for error reporting later

# Initial model training and evaluation
X_train, X_test, y_train, y_test, mop_train, mop_test = train_test_split(
    X, y, mop_ids, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}
gs = GridSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)
gs.fit(X_train, y_train)
model = gs.best_estimator_
print('Best parameters:', gs.best_params_)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.3f}, R2: {r2:.3f}')

# Train size optimisation
train_sizes = np.arange(0.5, 1.0, 0.05)
r2_scores = []
mse_scores = []
best_model = None
best_r2 = -np.inf
best_y_test = None
best_y_pred = None
best_size = None
best_params = None
best_mop_test = None

for size in train_sizes:
    X_train, X_test, y_train, y_test, mop_train, mop_test = train_test_split(
        X, y, mop_ids, test_size=1-size, random_state=42
    )
    gs = GridSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)
    print(f"Train size: {int(size*100)}%, MSE: {mse:.3f}, R2: {r2:.3f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_y_test = y_test
        best_y_pred = y_pred
        best_size = size
        best_params = gs.best_params_
        best_mop_test = mop_test

plt.figure()
plt.plot([int(s*100) for s in train_sizes], r2_scores, marker='o', label='R2')
plt.plot([int(s*100) for s in train_sizes], mse_scores, marker='s', label='MSE')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Score')
plt.title('Training Set Size Optimisation (XGBoost)')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
xgb.plot_importance(best_model)
plt.tight_layout()
plt.show()

# Final model on optimal split
X_train_final, X_test_final, y_train_final, y_test_final, mop_train_final, mop_test_final = train_test_split(
    X, y, mop_ids, test_size=1-best_size, random_state=42
)
best_model.fit(X_train_final, y_train_final)

y_train_pred = best_model.predict(X_train_final)
y_test_pred = best_model.predict(X_test_final)

mse_train = mean_squared_error(y_train_final, y_train_pred)
r2_train = r2_score(y_train_final, y_train_pred)
mae_train = mean_absolute_error(y_train_final, y_train_pred)

mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)
mae_test = mean_absolute_error(y_test_final, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for ax, y_true, y_pred, title, r2, mse, mae in zip(
    axes,
    [y_train_final, y_test_final],
    [y_train_pred, y_test_pred],
    ['Train Set', 'Test Set'],
    [r2_train, r2_test],
    [mse_train, mse_test],
    [mae_train, mae_test]
):
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_title(f'{title} Parity Plot')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))



error_df = pd.DataFrame({
    'true': y_test_final.values,
    'pred': y_test_pred,
    'mop_id': mop_test_final.values
})
error_df['abs_error'] = np.abs(error_df['true'] - error_df['pred'])
error_df['rel_error'] = error_df['abs_error'] / (np.abs(error_df['true']) + 1e-8)
worst = error_df.sort_values(by='abs_error', ascending=False)
best = error_df.sort_values(by='abs_error', ascending=True)


print("\nWorst predictions by absolute error:")
print(worst.head(10).to_string(index=False, float_format="%.3f"))


print("\nBest predictions by absolute error:")
print(best.head(10).to_string(index=False, float_format="%.3f"))


plt.tight_layout()
plt.show()

#Get error bars for data
#Look at basin hopping algorithm