import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('all_results_for_twa.csv')

# Drop rows with errors or missing values
df = df.replace('error', pd.NA).dropna()

# Convert numeric columns to float (skip filename and inner_atom)
for col in ['avg_metal_dist', 'inner_diameter', 'inner_volume', 'max_pore_size_diameter', 'max_pore_size_volume']:
    df[col] = df[col].astype(float)

# Choose target and features
target = 'max_pore_size_diameter'  # Change this to any geometric feature you want to predict

# Remove paired feature (diameter/volume) if present
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametric optimisation (GridSearchCV)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}
gs = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                  param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(X_train, y_train)
model = gs.best_estimator_
print('Best parameters:', gs.best_params_)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.3f}, R2: {r2:.3f}')

# Training set size optimisation for XGBoost
train_sizes = np.arange(0.5, 1.0, 0.05)
r2_scores = []
mse_scores = []
best_model = None
best_r2 = -np.inf
best_y_test = None
best_y_pred = None
best_size = None
best_params = None
for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size, random_state=42)
    gs = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                      param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
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

plt.figure()
plt.plot([int(s*100) for s in train_sizes], r2_scores, marker='o', label='R2')
plt.plot([int(s*100) for s in train_sizes], mse_scores, marker='s', label='MSE')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Score')
plt.title('Training Set Size Optimisation (XGBoost)')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance plot for best model
xgb.plot_importance(best_model)
plt.tight_layout()
plt.show()

# Parity plot for best configuration
print(f'Best R2: {best_r2:.3f} at train size {int(best_size*100)}%')
print('Best parameters:', best_params)
def parity_plot(y_true, y_pred, mse, r2):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot (Best R2)')
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMSE = {mse:.3f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.tight_layout()
    plt.show()

parity_plot(best_y_test, best_y_pred, mean_squared_error(best_y_test, best_y_pred), best_r2)
