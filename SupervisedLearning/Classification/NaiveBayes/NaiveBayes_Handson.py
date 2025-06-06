import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

dataframe = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Regression/LogisticRegression/diabetes.csv')
df = dataframe

'''print(df.head(10))
sections = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
for i in sections:
    if df[i].isnull().sum()>0:
        print("the null values are", df[i][df[i].isnull()])                                       
    else:
        pass
'''   
# Initialize an empty list


print(dataframe)
# Dictionary of sections
sections = {1:'Pregnancies', 2: 'Glucose', 3: 'BloodPressure', 4: 'SkinThickness',
            5: 'Insulin', 6: 'BMI', 7: 'DiabetesPedigreeFunction', 8: 'Age', 9: 'Outcome'}
sections2 = ['Pregnancies',  'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',  'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# Specify the row index

# Traverse all columns correctly
#sns.histplot(dataframe.Pregnancies)
#sns.displot(dataframe.BloodPressure)
#sns.displot(dataframe.Insulin)
#plt.show(block=False)

df_clean = df.copy()      # ← now df_clean is totally separate from df
'''for col in sections.values():
    df_clean[col] = df_clean[col].astype('float64')
    
    #casting back replacement values  
    #df_clean.at[idx, col] = int(df_clean[col].mean)
    
    
    Q1, Q3 = df_clean[col].quantile([.25, .75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    #building a boolean mask to all positions at once for the bad values
    mask = (df_clean[col] == 0) | (df_clean[col]>IQR) #requires seperate explicit parantheses
    #assigning the mean to all those positions at once 
    df_clean.loc[mask, col] = df_clean[col].mean()
    

    for idx in range(len(df_clean)):
        val = df_clean.at[idx, col]
        if (val == 0 or upper <  val or val < lower):        # or use (val < lower or val > upper)
            df_clean.at[idx, col] = df_clean[col].mean()
        else:
            pass'''

a, b, d = 0, [], []
for col in sections.values():
    df_clean[col] = df_clean[col].astype('float64')
    #Zscore = (x - u) / mew 
    for values in range(len(df_clean)):
        c = df_clean.at[values, col]
        a = a + c
    b.append(a)
    
    #x, j = 0, 2

for i in b:
    try:
        c = i/(len(df_clean))
        d.append(c)
    except ZeroDivisionError:
        c = None



        
        
        
        
        
        
        #std_dev = np.std(values)
        #print(values)
        #z.append(std_dev)
    #print("Standard deviation", std_dev)


print(d)
fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(data = dataNew, ax=ax)
plt.savefig('boxPlot.jpg')

#data cleaning is done
#sns.displot(dataframe.Pregnancies)
#sns.displot(dataframe.BloodPressure)
#sns.displot(dataframe.Insulin)
#plt.show()

df_clean = df.copy()
new_df = df_clean.copy()
#df_clean.to_csv('cleaned_diabetes_data.csv', index = False)

# Print only the values, not the entire list

# If you must print the full list once:
#print("\nFinal List:", a)  # Prints `a` only once


#training the naive bayes +
#model_gaussian_naive_bayes = GaussianNB()
#model_gaussian_naive_bayes.fit(X_train_resampled, y_train_resampled)

#model evaluation 
#y_predict_gaussian_naive_bayes = model_gaussian_naive_bayes.predict.(X_test)
#print(y_predict_gaussian_naive_bayes)
#print("Confusion Matrix")
#print(confusion_matrix(y_test, y_predict_gaussian_naive_bayes))
#print("Classification Report")
#print(classification_report(y_test, y_predict_gaussian_naive_bayes))