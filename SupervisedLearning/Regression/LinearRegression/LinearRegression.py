print("hello world")

#predicting house prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing #we used sklearn.dataesets for the extraction of datasets

#Now we are gonna assign a variable and give in the values
housing = fetch_california_housing()

#A small demostraion of how to load custom database sets for seaborn
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
print(df.head())
sns.histplot(data=df, x='MedHouseVal')
plt.show()
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal')
plt.show()

df.head()

housing

"""Learn about loading custom dataset
**USE YOUR BRAIN**

"""

print(housing.DESCR)

"""
***We can also check the values of any particular key in the 'housing' sklearn.utils._bunch.Bunch data structure. ***
---



"""

# taking (or) querying a particular attributes of an dataset
print(housing.feature_names) #most particularly - predictive attributes/independent variables

"""Tabular Structure: Both a data matrix for machine learning and a pivot table organize data in a tabular format with rows and columns.



**Rows as Entities:** In a pivot table, each row often represents a spec ific entity or combination of entities (e.g., sales for a particular product in a specific region). Similarly, in a machine learning data matrix, each row represents a single data sample (e.g., a house, a customer, a sensor reading).



**Columns as Attributes/Dimensions:** In a pivot table, columns can represent different attributes or dimensions along which you are summarizing data (e.g., product category, region, time period). In a machine learning data matrix, columns represent the predictive attributes or features of your data samples (e.g., size, number of bedrooms, age).



**Focus on Relationships**: Both structures aim to reveal relationships within the data. A pivot table helps you see how one variable changes with respect to others (e.g., how sales vary by region and product). A machine learning model learns the relationship between the predictive attributes (columns of the data matrix) and the target variable.

*And also somewhat like a correlation heatmap of seaborn.
Each specifying a correlaion and relationship as said. giving us some insights into the data we needed*
"""

print(housing.feature_names)
print(housing.target) #target variables are the ones which we got to predict from the feature variable

# **DATA PREPARATION**
#DATA-PREPARATION is the most crucial for machine learning
#creation of data frame
#machine learning pipelining that involves cleaning, transforming and organizing

dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names) #try labelling and lot other stuffs along with this
print(dataset)

#taking only the first 'n' rows of the dataset
dataset.head(n=6)
dataset.tail()


#adding new columns
dataset['Price'] = housing.target
dataset.head()

dataset['NoPrice'] = housing.target # Note: This re-assigns the 'NoPrice' column if it exists, or creates it.

#getting summary of the dataframe
dataset.info()

"""    Count: Number of non-null values.
    Mean: Average value.
    Std: Standard deviation, a measure of the amount of variation or dispersion.
    Min: Minimum value.
    25%: First quartile (25th percentile).
    50%: Median (50th percentile).
    75%: Third quartile (75th percentile).
    Max: Maximum value. *
    
"""

dataset.describe()

missing_values = dataset.isnull()
print(missing_values)
missing_count_per_column = missing_values.sum() # Corrected: .sum() directly on boolean DataFrame
print(missing_count_per_column)

#dropping the null values
dataset_without_missing = dataset.dropna()
print(dataset_without_missing)

#filling the null values
# Note: California housing dataset from sklearn usually doesn't have missing values.
# If it did, .mean() would need to be calculated on numeric columns only if there are non-numeric ones.
dataset_filled = dataset.fillna(dataset.mean(numeric_only=True)) # Added numeric_only=True for robustness
print(dataset_filled)

#for repleacing the table completely
# dataset_replaced = dataset.fillna(dataset.mean(numeric_only=True), inplace=True) # Added numeric_only=True
# print(dataset_replaced) # inplace=True makes the method return None, so dataset_replaced will be None.
# The original 'dataset' DataFrame would be modified.
# For clarity, it's often better to do:
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
print(dataset.head()) # Print dataset to see the effect


#for column values
# Corrected line: Use 'columns' instead of 'column' and provide a list of column indices or labels
# This line drops the column at index 1 (the second column)
# dataset_no_missing_column = dataset.drop(columns=dataset.columns[[1]])
# print(dataset_no_missing_column.head()) # Example usage

# Re-initializing dataset for clarity before splitting and modeling (as in the original script flow)
dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
dataset['Price'] = housing.target # This is the target variable for prediction
print(dataset.head())


housing_df = fetch_california_housing(as_frame=True).frame

# To get a DataFrame with only 'MedInc'
df_medinc_direct = housing_df[['MedInc']]
print(df_medinc_direct.head())

# To get a DataFrame with only 'HouseAge'
df_houseage_direct = housing_df[['HouseAge']]
print(df_houseage_direct.head())

print(housing_df.head(n=6))

#getting started with EDA(Exploratory Data Analysis)
#correlation matrix (expression of linear relationship)
print(dataset.corr()) # Print the correlation matrix
# sns.pairplot(dataset) # This can be very slow for datasets with many features.
# plt.show() # Show pairplot if uncommented

sns.boxplot(data=dataset) #boxplot to detect the outliers
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout() # Adjust layout
plt.savefig("boxplot.jpg")
plt.show()

"""**Avoiding the bias**

data-normalisation is one of the most important step of data-science
ensures fair comparisons betweeen features, especially when working with machine learning

Min-Max Scaling/Min-Max normalisation
used when we got to ensure that all features contribute equally to the model, regardless of their original scales
normalisation that scales the values of a features to a specific range(usually between 0 and 1)


Formula for min max scaling are as follows
Xnormalized = (X - X(min)) / (X(max) - X(min))


it is one of the pre-processing methods
"""

from sklearn.preprocessing import MinMaxScaler
#we assign a variable now
scaler = MinMaxScaler()
# Note: The following line normalizes the entire 'dataset', including the 'Price' (target) column.
# This 'normalized_data' is not explicitly used later for training the regression model in the script.
# The regression model is trained on 'X_train' which comes from an unscaled 'dataset'.
normalized_data = scaler.fit_transform(dataset)
print("Shape of normalized_data:", normalized_data.shape)


#splitting data into train and test sets are more of a crucial step in machine learning
#scikit-learn library provides a convinient stuff for train test split
features = dataset.iloc[:,:-1] # All columns except the last one ('Price')
#since mostly the last column is the target variable that our model wanna predict, we exclude those values to provide a proper and precise model
target = dataset.iloc[:, -1] # The last column ('Price')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
# X_train is unscaled features. scaler2 is fitted on this unscaled X_train.
x_train_norm = scaler2.fit_transform(X_train)#fits scaler into the training data dn transforms the training data accordingly
x_test_norm = scaler2.transform(X_test)#uses the same scaler to transform the test data. The scaler applies the transformation learneed from the training data to maimtain consistency

"""----

***Consistent scaling between the training and test sets is crucial for the model to make accurate predictions on new, unseen data. It ensures that the scaling characteristics learned from the training set are applied uniformly to the test set.***

In many machine learning algorithms, the scale of your features can significantly impact performance. For example, if one feature ranges from 0 to 1, and another ranges from 1000 to 10000, the algorithm might give more weight to the feature with larger values simply because of its magnitude, not necessarily its importance. Feature scaling aims to bring all your features to a similar range. Common methods include:

-----
"""

#model training
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score # r2_score is already here

#initializing linear regression model
regression = LinearRegression()

#Fitting the model to the training data
# IMPORTANT NOTE: Model is being trained on X_train (unscaled data)
regression.fit(X_train, y_train)

#retrieving slope values
slope_values = regression.coef_
print("Slope values (coefficients):", slope_values)

#retrieving the y-intercept (c) value
y_intercept = regression.intercept_
print("Y-intercept:", y_intercept)

"""----

Here, in this model we'll get 8 slope values and slope_values will contain an array of slopes corresponding to each feature in your dataset, and intercept_value will be a single value representing the y-intercept.
"""

#model prediction
# IMPORTANT NOTE: Predictions are being made on x_test_norm (scaled test data),
# while the model was trained on X_train (unscaled training data).
# This mismatch can lead to incorrect predictions and model evaluations.
# For a correct approach, the model should be trained on scaled training data (x_train_norm)
# if predictions are to be made on scaled test data (x_test_norm).
# Or, predict on unscaled X_test if trained on unscaled X_train.
# However, per instruction, existing code is not changed.
reg_pred = regression.predict(x_test_norm)

#calculating residuals
print("\nSample of y_test (actual values):")
print(y_test[:5].values)
print("\nSample of reg_pred (predicted values):")
print(reg_pred[:5])

residuals = y_test - reg_pred
print("\nSample of residuals:")
print(residuals[:5].values)


#visualizing the residuals
sns.displot(residuals, kind='kde')
plt.title('Distribution of Residuals (KDE)')
plt.xlabel('Residual Value')
plt.ylabel('Density')
plt.show()


# Assuming 'residuals' is the array of your model's residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins='auto') # 'auto' lets seaborn choose bins
plt.title('Distribution of Residuals (Histogram with KDE)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Try different numbers of bins:
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30) # Example: 30 bins
plt.title('Distribution of Residuals (30 bins)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

#**Model Evaluvation**
from sklearn.metrics import mean_absolute_error # already imported mean_squared_error, r2_score

#calculating the metrics
#mean square error and mean absolute error
mse = mean_squared_error(y_test, reg_pred)
mae = mean_absolute_error(y_test, reg_pred)
r2s = r2_score(y_test, reg_pred)  #variates in regressions
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2 score): {r2s}")


#calculating adjusted R-Square
# Assuming 'n_features' is the number of features in your model
n = len(y_test)  # Number of observations
# r2 = r2_score(y_test, reg_pred) # r2s is already calculated as r2_score
n_features = x_test_norm.shape[1] # or X_train.shape[1] as features are the same
if n - n_features - 1 == 0:
    adjusted_r2 = float('nan') # Avoid division by zero
    print("Warning: Cannot compute Adjusted R-squared due to n - k - 1 = 0.")
else:
    adjusted_r2 = 1 - ((1 - r2s) * (n - 1) / (n - n_features - 1))

print(f"Adjusted R-squared: {adjusted_r2:.4f}") # Increased precision for adjusted R2

#Calculating RMSE
# mse = mean_squared_error(y_test, reg_pred) # mse already calculated
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(mse) # mse already printed

#**SAVING THE TRAINING MODEL**
import pickle # pickle is imported here again, it's fine.
pickle.dump(regression, open('reg_model.pkl', 'wb'))
print("\nModel saved as reg_model.pkl")

"""The pickle.dump() function serializes the trained model (model) and writes it to the opened file (model_file). This effectively saves the model to a file.
open('reg_model.pkl', 'wb'):

This line opens a file named 'reg_model.pkl' in binary write mode ('wb'). The 'wb' mode is used for binary files, as pickling involves writing binary data.

----

reg_model.pkl file

When you save a machine learning model using pickle and name the file as "reg_model.pkl", the resulting file will contain the serialized version of the trained regression model. Here's what you can expect the "reg_model.pkl" file to have:
Serialized Model Parameters:

The serialized version of the trained regression model, including all the parameters, coefficients, and other information necessary to represent the model's state.
Model Type Information:

Information about the type of regression model (linear regression, ridge regression, etc.) and any specificd used for making predictions without having to retrain the model.
"""

#Sc

# --- Loading and using the pickled model --- #
print("\n# --- Loading and using the pickled model --- #")

# Load the model from the file
try:
    with open('reg_model.pkl', 'rb') as file:
        loaded_regression_model = pickle.load(file)
    print("Model loaded successfully from reg_model.pkl")

    # You can now use the loaded_model to make predictions.
    # Following the original script's pattern: predict on x_test_norm (scaled test data)
    # with the model trained on X_train (unscaled training data).
    loaded_model_predictions = loaded_regression_model.predict(x_test_norm)

    print("\nFirst 5 predictions using the loaded model:")
    print(loaded_model_predictions[:5])

    # Optionally, verify the loaded model by checking its parameters or re-evaluating
    print("\nParameters of the loaded model:")
    print("Coefficients:", loaded_regression_model.coef_)
    print("Intercept:", loaded_regression_model.intercept_)

    # Re-evaluate the loaded model (should give same results as original if no issues)
    mse_loaded = mean_squared_error(y_test, loaded_model_predictions)
    mae_loaded = mean_absolute_error(y_test, loaded_model_predictions)
    r2s_loaded = r2_score(y_test, loaded_model_predictions)
    rmse_loaded = np.sqrt(mse_loaded)

    print(f"\nMetrics for loaded model:")
    print(f"  Mean Squared Error (MSE): {mse_loaded}")
    print(f"  Mean Absolute Error (MAE): {mae_loaded}")
    print(f"  R-squared (R2 score): {r2s_loaded}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_loaded}")

    # Compare with original model's predictions and metrics (optional check)
    if np.allclose(reg_pred, loaded_model_predictions):
        print("\nPredictions from loaded model match original model's predictions.")
    else:
        print("\nWARNING: Predictions from loaded model DO NOT match original model's predictions.")

    if np.isclose(mse, mse_loaded) and np.isclose(r2s, r2s_loaded):
        print("Metrics (MSE, R2) from loaded model match original model's metrics.")
    else:
        print("WARNING: Metrics from loaded model DO NOT match original model's metrics.")

except FileNotFoundError:
    print("Error: 'reg_model.pkl' not found. Make sure the model was saved correctly.")
except Exception as e:
    print(f"An error occurred while loading or using the pickled model: {e}")