import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # For visualizing clusters if needed
from sklearn.preprocessing import PolynomialFeatures
import re
import warnings
import kagglehub
pd.set_option('display.max_columns', None)


# Download latest version
path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")

print("Path to dataset files:", path)

ds_raw = pd.read_csv(path + "/Housing.csv", header=0)

# check the loaded structure
print("Raw DataFrame Info:")
print(ds_raw.info())
print("\nRaw DataFrame Head:")
print(ds_raw.head(10))
print("\nColumn Names:", ds_raw.columns.tolist())
columns = ds_raw.columns.tolist()

warnings.filterwarnings('ignore')

ds = ds_raw.copy()

plt.figure(figsize=(15, 10))
sns.scatterplot(data=ds, x='area', y='price')
plt.grid(True, axis="y")

plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=ds, x='bedrooms', y='price')
plt.grid(True)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15, 10))
plt.grid(True)
sns.histplot(ds['price'], bins=60, kde=True)
plt.title('Price Distribution')
plt.show()

plt.figure(figsize=(15, 10))
plt.grid(True)
sns.histplot(ds['area'], bins=60, kde=True)
plt.title('Area Distribution')
plt.show()


# remove outliers
ds_encoded = ds.copy()

print("Removing outliers from price column...")

Q1 = ds_encoded['price'].quantile(0.25)
Q3 = ds_encoded['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers before removal
outliers_before = len(ds_encoded[(ds_encoded['price'] < lower_bound) | (ds_encoded['price'] > upper_bound)])
total_before = len(ds_encoded)

print(f"Price range before outlier removal: ${ds_encoded['price'].min():,.2f} - ${ds_encoded['price'].max():,.2f}")
print(f"Outliers detected: {outliers_before} ({outliers_before / total_before * 100:.1f}%)")
print(f"Outlier bounds: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

# Remove outliers
ds_encoded_clean = ds_encoded[(ds_encoded['price'] >= lower_bound) & (ds_encoded['price'] <= upper_bound)].copy()

print(
    f"Dataset size after outlier removal: {len(ds_encoded_clean)} ({len(ds_encoded_clean) / total_before * 100:.1f}% retained)")

# Update df_encoded to use the cleaned version
ds_encoded = ds_encoded_clean

# new price normal curve
plt.figure(figsize=(15, 10))
plt.grid(True)
sns.histplot(ds_encoded['price'], bins=60, kde=True)
plt.title('Price Distribution')
plt.show()

print("Removing outliers from area column...")

Q1 = ds_encoded['area'].quantile(0.25)
Q3 = ds_encoded['area'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers before removal
outliers_before = len(ds_encoded[(ds_encoded['area'] < lower_bound) | (ds_encoded['area'] > upper_bound)])
# total_before = len(ds_encoded)

print(f"Area range before outlier removal: {ds_encoded['area'].min():,.2f}m2 - ${ds_encoded['area'].max():,.2f}m2")
print(f"Outliers detected: {outliers_before} ({outliers_before / total_before * 100:.1f}%)")
print(f"Outlier bounds: {lower_bound:,.2f}m2 - {upper_bound:,.2f}m2")

# Remove outliers
ds_encoded_clean = ds_encoded[(ds_encoded['area'] >= lower_bound) & (ds_encoded['area'] <= upper_bound)].copy()

print(
    f"Dataset size after outlier removal: {len(ds_encoded_clean)} ({len(ds_encoded_clean) / total_before * 100:.1f}% retained)")


# Update df_encoded to use the cleaned version
ds_encoded = ds_encoded_clean

print("\nDataFrame Head without outliers:")
print(ds_encoded.head(10))

# new area normal curve
plt.figure(figsize=(15, 10))
plt.grid(True)
sns.histplot(ds_encoded['area'], bins=60, kde=True)
plt.title('Area Distribution')
plt.show()

# beed furnisinhg status
plt.figure(figsize=(15, 10))
sns.scatterplot(data=ds, x='furnishingstatus', y='price')
plt.grid(True, axis="y")

plt.xticks(rotation=90)
plt.show()

# ----------------

# categorical spectrum encoding
# One-hot encode the original raw column
ds_encoded = pd.get_dummies(ds_encoded, columns=['furnishingstatus'], prefix='furnish', drop_first=True)
ds_encoded[['furnish_semi-furnished', 'furnish_unfurnished']] = ds_encoded[['furnish_semi-furnished', 'furnish_unfurnished']].astype(int)

binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

for col in binary_columns:
    ds_encoded[col] = ds_encoded[col].map({"yes": 1, "no": 0})


# Convert numeric columns
for col in ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
    if col in ds_encoded.columns:
        ds_encoded[col] = pd.to_numeric(ds_encoded[col], errors='coerce')


print("\nDataFrame Head with Binary:")
print(ds_encoded.head(10))

# ---------------------------- models


features = ds_encoded.columns.tolist().copy()
features.pop(0)


df = ds_encoded[features + ['price']].copy()
X = df[features]
Y = df['price']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

lr = LinearRegression()
poly2 = PolynomialFeatures(degree=2, include_bias=False)

# liniar
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(Y_test, Y_pred_lr))
r2_lr = r2_score(Y_test, Y_pred_lr)

print(f"\nMODEL lr:")
print(f"Features used: {len(features)}")
print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R²: {r2_lr:.3f}")

plt.figure(figsize=(15, 10))
plt.scatter(Y_test, Y_pred_lr, alpha=0.6, color='green')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Linear\nMAE: {mae_lr:.0f}, R²: {r2_lr:.3f}')
plt.grid(True)
plt.show()


# poly 2
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

poly2_model = LinearRegression()
poly2_model.fit(X_train_poly2, Y_train)
Y_pred_poly2 = poly2_model.predict(X_test_poly2)

mae_poly2 = mean_absolute_error(Y_test, Y_pred_poly2)
rmse_poly2 = np.sqrt(mean_squared_error(Y_test, Y_pred_poly2))
r2_poly2 = r2_score(Y_test, Y_pred_poly2)

print(f"\nMODEL: Polynomial Regression (Degree 2)")
print(f"Original features: {len(features)} → Transformed features: {X_train_poly2.shape[1]}")
print(f"MAE: {mae_poly2:.2f}")
print(f"RMSE: {rmse_poly2:.2f}")
print(f"R²: {r2_poly2:.3f}")

# Plot Degree 2
plt.figure(figsize=(15, 10))
plt.scatter(Y_test, Y_pred_poly2, alpha=0.6, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Polynomial Regression (Degree 2)\nMAE: {mae_poly2:.0f}, R²: {r2_poly2:.3f}')
plt.grid(True)
plt.show()

