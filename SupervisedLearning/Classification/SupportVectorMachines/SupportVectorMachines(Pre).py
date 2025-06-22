import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import io 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()
print(type(df))


def data_exploration():
    
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    
    print(f"----------- DataFrame Description -----------\n{df.describe().to_string()}\n\n----------- DataFrame Shape -----------\n{df.shape}\n\n----------- DataFrame Info -----------\n{info_output}")

def imputations():
    #we have given some really good imputation methods in intricacies.md, do implement it later
    global imputed
    global df_transformed
    missing_values = df.columns[df.isnull().any()].tolist()
    if not missing_values:
        print("There are no values found")
    else:
        imputer  = IterativeImputer(max_iter = 10, random_state=50, initial_strategy='mean', imputation_order='ascending')
        df_imputed = imputer.fit(df) #fit the data first and then transform
        df_imputed_array = df_imputed.transform(df)
        df_transformed = pd.DataFrame(df_imputed_array, columns=df.columns, index=df.index)
        print("The values have been imputed")
        return df_transformed
    
    #print(type(df_transformed)) //dont forget to look into this implication
'''We did imputations way before oultier analysis because, the NaN values might mess up with the outlier handling 
essentially producing what we call as a pseudo-outlier'''

def manual_outlier_analysis():
    
    if df_transformed.isnull().any().any():
        print("Presence of missing values found. Review more")
    else:
        print("Continuing with outlier handling and analysis")
    
    #here the biggest problem with IQR analysis has been given down, Please do refer
    
    """
        Advanced Outlier Handling: A Guide to Moving Beyond the IQR Method

        The IQR (Interquartile Range) method is a great starting point for outlier detection, but it has critical limitations. This guide summarizes them and presents advanced alternatives.

        LIMITATIONS OF THE IQR METHOD:
        -----------------------------
        1.  It is Univariate: It only checks one feature at a time and is blind to outliers that have a strange *combination* of otherwise normal values (e.g., a 10-year-old with a very high income).

        2.  The "1.5" Multiplier is a Heuristic: The standard 1.5xIQR rule is a general guideline, not a universal law. It may be too aggressive or too lenient for your specific data distribution.

        3.  Fails on Multi-Modal Data: On data with multiple peaks (bimodal), the IQR can become so large that it "masks" outliers, failing to identify them.

        HOW TO OVERCOME THEM:
        ---------------------
        The key is to move from a simple statistical rule to more advanced, model-based methods that can see the bigger picture.

        1.  For Multivariate Outliers (The Biggest Limitation):
            Use algorithms that consider all features at once.
            - Isolation Forest: A fast and effective machine learning model that "isolates" anomalies. It's excellent for high-dimensional data.
            - DBSCAN: A density-based clustering algorithm that finds points in low-density regions and flags them as outliers. Perfect for complex data shapes.
            - Local Outlier Factor (LOF): A clever algorithm that compares the local density of a point to its neighbors to find anomalies.

        2.  General Best-Practice Workflow:
            a. Visualize First: Always plot a histogram or boxplot to understand your data's shape before applying any rule.
            b. Start with IQR: Use it for a quick, robust check on individual features.
            c. Upgrade When Needed: If you suspect relationships between features are important or if your data has a complex structure, use a multivariate method like Isolation Forest or DBSCAN for a more accurate analysis.

    """
    
    


data_exploration(df)
imputations(df)