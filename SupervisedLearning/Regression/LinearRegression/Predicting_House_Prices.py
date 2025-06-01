#predicting house prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# --- 0. Initial Setup and Style ---
# Apply a visually appealing style for plots
plt.style.use('seaborn-v0_8-whitegrid')
print("Hello Darshan! Let's predict some house prices with enhanced visuals and corrections.")

# --- 1. Load Dataset ---
# We use sklearn.datasets for the extraction of datasets
housing = fetch_california_housing()

# Create a Pandas DataFrame for easier manipulation and analysis
# housing.data contains the features, housing.feature_names the column names
df = pd.DataFrame(housing.data, columns=housing.feature_names)
# housing.target contains the target variable (median house value)
df['MedHouseVal'] = housing.target # This is the actual target variable we want to predict

print("\n--- Initial DataFrame Head ---")
print(df.head())

# --- 2. Initial Data Exploration & Visualization ---
print("\n--- Initial Data Visualizations ---")

# Histogram of the target variable (Median House Value)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MedHouseVal', kde=True, bins=50)
plt.title('Distribution of Median House Value (MedHouseVal)')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot of Median Income vs. Median House Value
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', alpha=0.5)
plt.title('Median Income (MedInc) vs. Median House Value (MedHouseVal)')
plt.xlabel('Median Income (tens of thousands of $)')
plt.ylabel('Median House Value ($100,000s)')
plt.tight_layout()
plt.show()

# --- 3. Understand the Dataset Structure ---
print("\n--- Dataset Description (DESCR) ---")
# The DESCR attribute provides a detailed description of the dataset
print(housing.DESCR)

print("\n--- Feature Names (Predictive Attributes) ---")
# These are the independent variables used for prediction
print(housing.feature_names)

print("\n--- Target Variable Name (What we predict) ---")
# This is the dependent variable
print(housing.target_names) # Usually ['MedHouseVal'] for this dataset

# --- 4. Data Preparation & Cleaning ---
# Re-assigning 'dataset' for clarity in this section, though 'df' is already our main DataFrame.
# We'll use 'df' consistently from now on.
# PITFALL ADDRESSED: Removed the redundant 'NoPrice' column from the original script.
# It was identical to 'MedHouseVal' (which was named 'Price' in the original script's 'dataset').
# df already has 'MedHouseVal' as the target.

print("\n--- DataFrame Info (Data Types and Non-Null Counts) ---")
df.info()
# This helps identify data types and potential missing values.

print("\n--- DataFrame Descriptive Statistics ---")
# Provides summary statistics for numerical columns (mean, std, min, max, quartiles)
print(df.describe())

print("\n--- Checking for Missing Values ---")
# PITFALL ADDRESSED: Corrected how missing values are summed.
# Original: missing_values.isnull().sum() on a boolean DataFrame.
# Correct: df.isnull().sum()
missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts)
if missing_counts.sum() == 0:
    print("No missing values found in this dataset. (This is typical for sklearn's California Housing dataset)")
else:
    print(f"Total missing values found: {missing_counts.sum()}")
    # If there were missing values, strategies like dropping or imputing would be applied here.
    # Example: df.dropna(inplace=True) # To drop rows with any missing values
    # Example: df.fillna(df.mean(numeric_only=True), inplace=True) # To fill with mean (for numeric columns)
    # PITFALL NOTE: 'inplace=True' modifies the DataFrame directly and returns None.
    # So, `dataset_replaced = dataset.fillna(..., inplace=True)` would make `dataset_replaced` None.
    # Better: `df.fillna(df.mean(numeric_only=True), inplace=True)` or `df = df.fillna(df.mean(numeric_only=True))`

# --- 5. Exploratory Data Analysis (EDA) - Deeper Dive ---
print("\n--- Exploratory Data Analysis (EDA) ---")

# Correlation Matrix Heatmap
# This shows the linear relationship between variables.
plt.figure(figsize=(12, 10))
# PITFALL ADDRESSED: Using a heatmap for correlation is much more visual than just printing the matrix.
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features and Target')
plt.tight_layout()
plt.show()

# Pairplot (can be computationally expensive for many features)
# It plots pairwise relationships in a dataset.
# PITFALL ADDRESSED: Commented out by default due to potential performance issues.
# Suggesting a more targeted pairplot if needed.
# print("\nGenerating Pairplot (can be slow)...")
# sns.pairplot(df, vars=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'MedHouseVal'], diag_kind='kde', corner=True)
# plt.suptitle('Pairplot of Selected Features and Target', y=1.02)
# plt.show()
print("Skipping full pairplot for efficiency. Consider plotting specific pairs if needed.")
# Example for a few key features including the target:
plt.figure()
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']], diag_kind='kde', corner=True)
plt.suptitle('Pairplot of Key Features and Target', y=1.02)
plt.tight_layout()
plt.show()


# Boxplot to detect outliers for all features
# PITFALL ADDRESSED: Rotated x-axis labels for better readability.
plt.figure(figsize=(15, 8))
sns.boxplot(data=df)
plt.title('Boxplot of All Features (to identify potential outliers)')
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout()
plt.show()
# Note: Outlier treatment (e.g., capping, removal, transformation) would be a subsequent step
# if outliers are deemed problematic for the chosen model.

# --- 6. Feature Scaling and Data Splitting ---
# The 'Price' column in the original script is 'MedHouseVal' in our 'df'.
features = df.drop('MedHouseVal', axis=1) # X: All columns except the target
target = df['MedHouseVal']             # y: Only the target column

# Splitting data into training and testing sets
# This is crucial to evaluate the model's performance on unseen data.
# test_size=0.2 means 20% of data for testing, 80% for training.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Feature Scaling
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
# MinMaxScaler (shown for illustration in original script, but StandardScaler is used for the model)
# scales features to a given range, usually [0, 1].
# PITFALL ADDRESSED: The original script had a MinMaxScaler applied to the whole dataset (including target)
# but this scaled data wasn't used for the LinearRegression model.
# We will focus on StandardScaler which is correctly applied here.

# Illustrative Min-Max Scaling (not used for the final model in this script)
# scaler_minmax = MinMaxScaler()
# X_train_minmax_scaled_example = scaler_minmax.fit_transform(X_train)
# X_test_minmax_scaled_example = scaler_minmax.transform(X_test)
# print("\nExample of MinMax Scaled X_train (first 5 rows):")
# print(pd.DataFrame(X_train_minmax_scaled_example, columns=X_train.columns).head())


# Using StandardScaler for the model
# 1. Initialize the scaler
feature_scaler = StandardScaler() # Renamed from 'scaler2' for clarity

# 2. Fit the scaler on the training data ONLY and then transform it.
#    Fitting learns the mean and standard deviation from the training data.
x_train_scaled = feature_scaler.fit_transform(X_train)

# 3. Transform the test data using the SAME scaler fitted on the training data.
#    This prevents data leakage from the test set into the training process.
x_test_scaled = feature_scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier inspection (optional)
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=X_train.columns)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=X_test.columns)

print("\n--- Scaled Training Data (StandardScaler) Head ---")
print(x_train_scaled_df.head())


# --- 7. Model Training ---
# We'll use Linear Regression.
regression_model = LinearRegression() # Renamed from 'regression' for clarity

# PITFALL ADDRESSED & CORRECTED:
# Original script trained on X_train (unscaled) but predicted on x_test_norm (scaled).
# This is a mismatch. The model should be trained on the same scale of data it will predict on.
# CORRECTED: Train the model on the SCALED training data.
print("\n--- Training the Linear Regression Model on SCALED data ---")
regression_model.fit(x_train_scaled, y_train)

# Retrieving model parameters
slope_values = regression_model.coef_
y_intercept = regression_model.intercept_

print(f"\nSlope values (coefficients) for each feature: {slope_values}")
print(f"Y-intercept (c) value: {y_intercept}")
print("\nFeature names corresponding to coefficients:")
for feature, coef in zip(X_train.columns, slope_values):
    print(f"  {feature}: {coef:.4f}")

# --- 8. Model Prediction ---
# PITFALL ADDRESSED & CORRECTED:
# Predict on the SCALED test data, consistent with how the model was trained.
print("\n--- Making Predictions on SCALED Test Data ---")
y_pred = regression_model.predict(x_test_scaled)

# Display a few actual vs. predicted values
results_df = pd.DataFrame({'Actual_MedHouseVal': y_test, 'Predicted_MedHouseVal': y_pred})
print("\n--- Sample of Actual vs. Predicted Median House Values ---")
print(results_df.head(10))

# --- 9. Residual Analysis ---
# Residuals are the differences between actual and predicted values.
residuals = y_test - y_pred
print("\n--- Sample of Residuals ---")
print(residuals.head())

print("\n--- Visualizing Residuals ---")
# Distribution of Residuals (Histogram/KDE)
# Ideally, residuals should be normally distributed around zero.
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=50)
plt.title('Distribution of Residuals (Actual - Predicted)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.axvline(residuals.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean Residual: {residuals.mean():.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# Scatter plot of Predicted Values vs. Residuals
# This helps check for homoscedasticity (constant variance of residuals).
# Ideally, there should be no clear pattern; points should be randomly scattered around y=0.
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Predicted Values vs. Residuals')
plt.xlabel('Predicted Median House Value')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# Scatter plot of Actual Values vs. Predicted Values
# Points should ideally fall along a 45-degree line.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
plt.title('Actual vs. Predicted Median House Values')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.tight_layout()
plt.show()


# --- 10. Model Evaluation ---
print("\n--- Model Evaluation Metrics ---")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) # R-squared
rmse = np.sqrt(mse)          # Root Mean Squared Error

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# Calculating Adjusted R-Square
n = len(y_test)  # Number of observations in the test set
k = x_test_scaled.shape[1]  # Number of features (predictors)

# PITFALL ADDRESSED: Added a check to prevent division by zero if n-k-1 is zero or negative.
if n - k - 1 <= 0:
    adjusted_r2 = float('nan')
    print("Warning: Adjusted R-squared cannot be reliably computed (n - k - 1 <= 0).")
else:
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R-squared: {adjusted_r2:.4f}")


# --- 11. Saving the Trained Model and the Scaler ---
# It's crucial to save not just the model, but also the scaler that was used to preprocess the training data.
# This allows you to apply the exact same transformation to new, unseen data before making predictions.
model_filename = 'regression_model.pkl'
scaler_filename = 'feature_scaler.pkl'

print(f"\n--- Saving the Model to {model_filename} and Scaler to {scaler_filename} ---")
with open(model_filename, 'wb') as file:
    pickle.dump(regression_model, file)
print(f"Model saved successfully as {model_filename}")

with open(scaler_filename, 'wb') as file:
    pickle.dump(feature_scaler, file)
print(f"Scaler saved successfully as {scaler_filename}")

# --- 12. Loading and Using the Pickled Model and Scaler ---
print("\n--- Loading and Using the Pickled Model and Scaler ---")

try:
    # Load the scaler first
    with open(scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)
    print(f"Scaler loaded successfully from {scaler_filename}")

    # Load the model
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Model loaded successfully from {model_filename}")

    # Now, let's simulate making predictions on new data (using our x_test_scaled for demonstration)
    # If you had new, raw data (e.g., new_X_raw), you would first scale it:
    # new_X_scaled = loaded_scaler.transform(new_X_raw)
    # loaded_model_predictions_new = loaded_model.predict(new_X_scaled)

    # For demonstration, predict on the original x_test_scaled again with the loaded model
    loaded_model_predictions = loaded_model.predict(x_test_scaled)

    print("\nFirst 5 predictions using the loaded model (on x_test_scaled):")
    print(loaded_model_predictions[:5])

    # Verify the loaded model by checking its parameters or re-evaluating
    print("\nParameters of the loaded model:")
    print(f"  Coefficients: {loaded_model.coef_}")
    print(f"  Intercept: {loaded_model.intercept_}")

    # Re-evaluate the loaded model (should give same results as original)
    mse_loaded = mean_squared_error(y_test, loaded_model_predictions)
    r2_loaded = r2_score(y_test, loaded_model_predictions)

    print(f"\nMetrics for loaded model (on x_test_scaled):")
    print(f"  Mean Squared Error (MSE): {mse_loaded:.4f}")
    print(f"  R-squared (R2 score): {r2_loaded:.4f}")

    # Compare with original model's predictions and metrics for sanity check
    if np.allclose(y_pred, loaded_model_predictions):
        print("\nSUCCESS: Predictions from loaded model match original model's predictions.")
    else:
        print("\nWARNING: Predictions from loaded model DO NOT match original model's predictions.")

    if np.isclose(mse, mse_loaded) and np.isclose(r2, r2_loaded):
        print("SUCCESS: Metrics (MSE, R2) from loaded model match original model's metrics.")
    else:
        print("WARNING: Metrics (MSE, R2) from loaded model DO NOT match original model's metrics.")

except FileNotFoundError:
    print(f"Error: Model ({model_filename}) or Scaler ({scaler_filename}) file not found. Make sure they were saved correctly.")
except Exception as e:
    print(f"An error occurred while loading or using the pickled model/scaler: {e}")

print("\n--- End of Enhanced Script ---")
