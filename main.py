import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import kagglehub
import warnings

# --- SETUP ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Download
path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
ds_raw = pd.read_csv(path + "/Housing.csv", header=0)
ds = ds_raw.copy()

# --- 1. OUTLIER REMOVAL (Price Cutoff) ---
# Filter Area
ds = ds[ds['area'] < 14000]

# Filter Top 20% Expensive Prices
max_price = ds['price'].max()
cutoff_limit = max_price * 0.80

rows_before = len(ds)
ds = ds[ds['price'] <= cutoff_limit]
print(f"Dataset clean. Removed {rows_before - len(ds)} high-end outliers.")
print(f"Max Price in dataset: ${ds['price'].max():,.0f}")

# --- 2. ENCODING ---
binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_columns:
    ds[col] = ds[col].map({"yes": 1, "no": 0})

# ds['car_need'] = ds['parking'] * ds['mainroad']
ds['rooms_floors'] = ds['bedrooms'] * ds['stories']
ds['area_pref'] = ds['area'] * ds['prefarea']
ds['bed_area'] = ds['bedrooms'] * ds['area']

# Quality Score
ds['quality_score'] = (ds['mainroad'] + ds['guestroom'] + ds['basement'] +
                       ds['hotwaterheating'] + ds['airconditioning'] + ds['prefarea'])

# --- ONE-HOT ENCODING (Categorical Treatment) ---
# NOW we convert these to categories. The original 'parking' and 'stories' columns are dropped here.
ds = pd.get_dummies(ds, columns=['furnishingstatus', 'stories', 'parking'], prefix=['furnish', 'story', 'park'], drop_first=True)

# Ensure they are integers
new_cols = [c for c in ds.columns if 'furnish_' in c or 'story_' in c or 'park_' in c]
ds[new_cols] = ds[new_cols].astype(int)

# Keep log_area as a FEATURE (helps inputs), but Target will be RAW
ds['log_area'] = np.log1p(ds['area'])

# --- 3. PREPARE RAW TARGET ---
# We Drop 'price' from X, but keep it as Y (No Log Transform on Y)
X = ds.drop(columns=['price', 'area'])
Y = ds['price']  # <--- RAW DOLLARS

# --- 4. POLYNOMIALS ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
X_poly = X_poly.loc[:, ~X_poly.columns.duplicated()]

# --- 5. SPLIT & SCALE ---
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# Scale Inputs (X)
scaler_x = RobustScaler()
X_train_scaled = pd.DataFrame(scaler_x.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler_x.transform(X_test), columns=X_test.columns)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=120)
selector.fit(X_train_scaled, Y_train)

cols_idx = selector.get_support(indices=True)
selected_cols = X_train_scaled.columns[cols_idx]
X_train_sel = X_train_scaled[selected_cols]
X_test_sel = X_test_scaled[selected_cols]

print(f"Features used: {X_train_sel.shape[1]}")

# --- 6. MODELING (NO LOG TARGET) ---

# We need a model that can handle raw dollar amounts.
# We wrap the Stacking Regressor in a TransformedTargetRegressor with StandardScaler.
# This scales the price to (Mean=0, Std=1) for training, then converts back to Dollars for prediction.
# It is LINEAR (Value friendly), not LOGARITHMIC.

# A. Base Models
ridge = Ridge(alpha=10)
gbr = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    max_features='sqrt',
    random_state=42
)

# B. The Stack
stack = StackingRegressor(
    estimators=[('ridge', ridge), ('gbr', gbr)],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

# C. The Linear Scaler Wrapper (Crucial for Values)
model = TransformedTargetRegressor(
    regressor=stack,
    transformer=StandardScaler() # <--- Linear scaling, not Log
)

print("Training on Dollar values...")
model.fit(X_train_sel, Y_train)

# --- 7. EVALUATION (Directly in Dollars) ---
y_pred = model.predict(X_test_sel)

# No inversion needed, Y_train and y_pred are already in Dollars
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r2 = r2_score(Y_test, y_pred)

print("\n" + "="*40)
print(f" FINAL RESULTS (Raw Values)")
print("="*40)
print(f"R2 Score:  {r2:.4f}")
print(f"MAE:       ${mae:,.2f}")
print(f"RMSE:      ${rmse:,.2f}")
print("-" * 40)

# --- 8. PLOT ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_test, y=y_pred, alpha=0.6, color='navy')

# Perfect Line
min_val = min(Y_test.min(), y_pred.min())
max_val = max(Y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.title(f'Prediction on Prices\nR2: {r2:.3f}')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.grid(True)
plt.ticklabel_format(style='plain', axis='both') # Disable scientific notation
plt.show()