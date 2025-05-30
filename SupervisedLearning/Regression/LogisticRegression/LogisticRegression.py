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
plt.savefig('correlation-coeff.jpg')
plt.show()
