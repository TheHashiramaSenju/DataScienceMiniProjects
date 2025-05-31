import seaborn as sns 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv("SupervisedLearning/Regression/LogisticRegression/diabetes.csv")

print(dataframe.head(20)) #we observe the data-structures here
print()
 
print(dataframe.shape) #observe the values and the types 
print() 

print(dataframe.corr)
print()

a = plt.figure(figsize=(15,15))
print(a)
print()

ax = sns.heatmap(dataframe.corr(), annot=True)
#plt.savefig('correlation-coeff.jpg')
plt.show()

print()
print(dataframe.shape)
sns.displot(dataframe.Pregnancies)
sns.displot(dataframe.BloodPressure)
sns.displot(dataframe.Insulin)
plt.show()

#imputing and replacing the missing values with required statistical methods
dataframe['Insulin'] = dataframe['Insulin'].replace(0, dataframe['Insulin'].median())
dataframe['Pregnancies'] = dataframe['Pregnancies'].replace(0, dataframe['Pregnancies'].mean())
dataframe['Glucose'] = dataframe['Glucose'].replace(0, dataframe['Glucose'].mean())
dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0, dataframe['BloodPressure'].mean())
dataframe['SkinThickness'] = dataframe['SkinThickness'].replace(0, dataframe['SkinThickness'].median())
dataframe['BMI'] = dataframe['BMI'].replace(0, dataframe['BMI'].mean())
dataframe['DiabetesPedigreeFunction'] = dataframe['DiabetesPedigreeFunction'].replace(0, dataframe['DiabetesPedigreeFunction'].median())
dataframe['Age'] = dataframe['Age'].replace(0, dataframe['Age'].median())

#know where to use which statistical method

#Now let us do outlier handling 
## Splitting the data into input features (X) and target value (y)
X = dataframe.drop(columns='Outcome', axis=1)
y = dataframe['Outcome']

fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(data = X, ax=ax)
plt.savefig('boxPlot.jpg')

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)

# Filter dataset to remove outliers
X_outlier_detection = X[mask]
y_outlier_detection = y[mask] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_outlier_detection)

#handling these outliers 
X_scaled.reset_index(drop=True, inplace=True)
y_outlier_detection.reset_index(drop = True, inplace = True)

#quantile based filtering 
q = X_scaled['Insulin'].quantile(.95)
mask = X_scaled['Insulin'] < q
dataNew = X_scaled[mask]
y_outlier_detection = y_outlier_detection[mask]

#visualization with boxplot
fig, ax = plt.subplots(figsize=(15, 15))
sns.boxplot(data=dataNew, ax=ax)

