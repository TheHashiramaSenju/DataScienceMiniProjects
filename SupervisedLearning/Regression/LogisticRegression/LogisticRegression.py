import seaborn as sns 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
dataframe = pd.read_csv("SupervisedLearning/Regression/LogisticRegression/diabetes.csv")

# Exploratory Data Analysis
print(dataframe.head(20)) 
print("\nShape:", dataframe.shape) 
print("\nCorrelation:\n", dataframe.corr())

# Visualization
plt.figure(figsize=(15,15))
sns.heatmap(dataframe.corr(), annot=True)
plt.show()

# Distribution plots
sns.displot(dataframe.Pregnancies)
sns.displot(dataframe.BloodPressure)
sns.displot(dataframe.Insulin)
plt.show()

# Handling Missing Values (Replacing zeros with mean/median)
columns_to_replace = {
    'Insulin': 'median', 'Pregnancies': 'mean', 'Glucose': 'mean',
    'BloodPressure': 'mean', 'SkinThickness': 'median', 'BMI': 'mean',
    'DiabetesPedigreeFunction': 'median', 'Age': 'median'
}
for col, method in columns_to_replace.items():
    dataframe[col] = dataframe[col].replace(0, getattr(dataframe[col], method)())

# Splitting Data
X = dataframe.drop(columns='Outcome', axis=1)
y = dataframe['Outcome']

# Boxplot for Outlier Detection
plt.figure(figsize=(15,15))
sns.boxplot(data=X)
plt.show()

# Removing Outliers using IQR (Corrected)
mask = pd.Series(True, index=X.index)  # Start with all True values

for col in X.columns:
    Q1, Q3 = X[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask &= (X[col] >= lower_bound) & (X[col] <= upper_bound)

# Apply the mask to both X and y
X = X[mask]
y = y.loc[mask]  # Ensuring proper indexing

# Standardization (Converting back to DataFrame)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Reset indices to avoid alignment issues in further processing
X_scaled.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Filtering Insulin values based on quantile
q = X_scaled['Insulin'].quantile(.95)
mask = X_scaled['Insulin'] < q
X_scaled = X_scaled[mask]
y = y.loc[mask]  # Ensure alignment

# Boxplot after handling outliers
plt.figure(figsize=(15,15))
sns.boxplot(data=X_scaled)
plt.show()

# Splitting Data into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# Handling Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nResampled Class Distribution:\n", pd.Series(y_train_resampled).value_counts())

# Logistic Regression Model Training
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_resampled, y_train_resampled)

# Model Prediction & Evaluation
y_predictions = logistic_regression_model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_predictions))

target_names = ['Non-Diabetic', 'Diabetic']
print("\nClassification Report:\n", classification_report(y_test, y_predictions, target_names=target_names))

# Saving the Model (Fixed)
with open("classification_model.pkl", "wb") as model_file:
    pickle.dump(logistic_regression_model, model_file)

# Loading the Saved Model & Making Predictions
with open("classification_model.pkl", "rb") as model_file:
    classification_model = pickle.load(model_file)

print("\nSample Predictions:", classification_model.predict(X_test[:5]))
