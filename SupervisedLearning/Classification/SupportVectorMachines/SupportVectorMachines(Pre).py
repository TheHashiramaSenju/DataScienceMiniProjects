import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import io 
from sklearn.impute import IterativeImputer


dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()

def data_exploration():
    
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    
    print(f"----------- DataFrame Description -----------\n{df.describe().to_string()}\n\n----------- DataFrame Shape -----------\n{df.shape}\n\n----------- DataFrame Info -----------\n{info_output}")

def imputations():
    #we have given some really good imputation methods in intricacies.md, do implement it later
    global imputed
    missing_values = df.columns[df.isnull().any()].tolist()
    if not missing_values:
        print("There are no values found")
    else:
        imputer  = IterativeImputer(max_iter = 10, random_state=50, initial_strategy='mean', imputation_order='ascending')
        df_imputed = imputer.fit(df) #fit the data first and then transform
        df_imputed_array = df_imputed.transform(df)
        df_transformed = pd.DataFrame(df_imputed_array, columns=df, index=df.index)
        print("The values have been imputed")
        return df_transformed

'''We did imputations way before oultier analysis because, the NaN values might mess up with the outlier handling 
essentially producing what we call as a pseudo-outlier'''

def outlier_analysis():
    pass


data_exploration()
imputations()