import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataframe = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Regression/LogisticRegression/diabetes.csv')
df = dataframe

print(df.head(10))
sections = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
for i in sections:
    if df[i].isnull:
        print("the null values are",df[i])


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