import seaborn as sns 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

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
