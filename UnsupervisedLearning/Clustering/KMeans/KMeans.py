#K - Means clustering
#we try to do customer segmenting 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
import io
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from pyculiarity import detect_ts


dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/UnsupervisedLearning/Clustering/KMeans/data.csv', delimiter=',', encoding='ISO-8859-1')
df = dataset.copy()

def data_exploration(df_to_explore):
    info_buffer = io.StringIO()
    df_to_explore.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    print(f"----------- DataFrame Description -----------\n{df_to_explore.describe().to_string()}\n\n----------- DataFrame Shape -----------\n{df_to_explore.shape}\n\n----------- DataFrame Info -----------\n{info_output}\n\n---------------------DataHead--------------------\n{df_to_explore.head(30)}\n\n---------------------DataTail--------------------\n{df_to_explore.tail(30)}")
    #since this is a time-series data, we analyzr the datas
    
data_exploration(df)

def data_handling(df_to_impute):
    
    missing_values = df_to_impute.columns[df_to_impute.isnull().any()].tolist()
    if not missing_values:
        print("No missing values, proceeding to next steps")
        df_corrected = df_to_impute
    if missing_values:
        df_corrected = df_to_impute.dropna()
        print(df_corrected)
    
    df_corrected = df_corrected[df_corrected['Quantity'] > 0]
    df_corrected[df_corrected['UnitPrice'] < 0].shape[0]
    
    if df_corrected :
        print("No negative values")
    if not df_corrected : 
        df_corrected.drop(df_corrected[df_corrected['Quantity'] < 0].index, inplace = True)
        return df_corrected
    
    #converting date-time from object to actual date-time for the model to process the data 
    df_corrected['InvoiceDate'] = pd.to_datetime(df_corrected['InvoiceDate'])
    return df_corrected

def advanced_data_handling(df_corrected):

    #date_range
    date_range = df_corrected["InvoiceDate"]
    
    #maximum and minimum date
    max_date = date_range.max()
    min_date = date_range.min()

    #date measures    
    print(f"Data spans from: {min_date}")
    print(f"Data spans to: {max_date}")
    print(f"Total days: {(max_date - min_date).days}")
    print(f"Total years: {(max_date - min_date).days / 365:.1f}")

    return max_date, min_date

def build_customer_intelligence(df_corrected, reference_date=None):
    
    if reference_date is None:
        reference_date = df_corrected["InvoiceDate"].max()
    
    print(f"Using reference date: {reference_date}")
    
    #now aggreagating and grouping the functions together
    
    customer_features = df_corrected.groupby('CustomerID').agg({
        'InvoiceDate' : ['count', 'max', 'min'],
        'sales' : ['sum', 'mean', 'std'],
        'Quantity' : ['sum', 'mean'],
        'InvoiceNo' : 'nunique',
        'StockCode' : 'nunique',
        'Country' : lambda x : x.mode()[0] if not x.mode().empty else 'Unknown'
    }).round(2)
    
    customer_features.columns = [
        'total_transactions', 'last_purchase', 'first_purchase',
        'total_spent', 'avg_order_value', 'spending_consistency',
        'total_quantity', 'avg_quantity_per_transaction',
        'unique_orders', 'product_diversity', 'primary_country'
    ]
    
    customer_features['recency_days'] = (reference_date - customer_features['last_purchase']).dt.days
    customer_features['customer_lifespan_days'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
    customer_features['avg_days_between_orders'] = customer_features['customer_lifespan_days'] / (customer_features['unique_orders'] - 1)
    customer_features['avg_days_between_orders'] = customer_features['avg_days_between_orders'].fillna(0)
    
    # Handle consistency (coefficient of variation)
    customer_features['spending_consistency'] = customer_features['spending_consistency'].fillna(0)
    customer_features['consistency_score'] = 1 / (1 + customer_features['spending_consistency'] / customer_features['avg_order_value'])
    customer_features['consistency_score'] = customer_features['consistency_score'].fillna(0.5)
    
    return customer_features


def feature_engineering(df_corrected):
    
    df_corrected['sales'] = df_corrected['Quantity'] * df_corrected['UnitPrice'] #pandas work in a very amazing manner, think about multiplying 2 columns this easily, Just WOWW --> vectorisation
    df_sales = df_corrected.groupby('CustomerID')['sales'].sum().reset_index #find why particularly sales, and why to group by and what is the use of this, how it help our data and predictions
    df_transactions = df_corrected.groupby('CustomerID')['InvoiceNo'].count().reset_index #see the SVM node side of this
    df_corrected['LastTransaction'] = (df_corrected['InvoiceDate'].max() - df_corrected['InvoiceDate']).dt.days # what are these .dt.days
    df_date = df_corrected.groupby(['CustomerID', 'Country'])['LastTransaction'].max().reset_index()
    
    #table merging 
    merge_table = pd.merge(df_date, df_transactions, how='inner', on="CustomerID") #revise joins --> merge function accepts only 2 dataframes at a time and not 3 
    new_df = pd.merge(merge_table, df_sales, how='inner', on='CustomerID') #why only inner
    return df_sales, df_transactions, df_date, merge_table, new_df





'''def handling_outlier(new_df, df_sales, df_transactions, df_date, merge_table):
    
    #visualizing the prepared data
    plt.figure(figsize=(30, 20))
    plt.xticks(rotation = 45, ha = 'right', fontsize = 10)
    plt.tight_layout()
    sns.boxplot(data=new_df[['InvoiceNo', 'LastTransaction', 'sales']]) #why 2 brackets now, what are they asking me to do?
    plt.show()
    
    #handling the outliers
    
    #handling time series data
    print(df_date)
    df_test1 = df_date.copy()
    df_test1['Time'] = pd.to_datetime(df_test1)
    df_test1['Timestamp'] = df_test1['Time'].apply(lambda x: x.timestamp())
    correction_data = df_test1[['Timestamp', 'LastTransaction']]
    correction_data2 = df_test1[['Timestamp', 'LastTransaction']].rename(columns={'LastTransaction' : 'value'})
    
    detected_outliers = detect_ts(
        correction_data2, 
        max_anoms=0.5,
        alpha=0.05,
        direction='both',
        only_last=None,
        longterm=True #play with values
    )
    print(detected_outliers) '''


def advanced_feature_engineering(df_corrected):
    #Recency = Today - Last Purchase Date
    #Frequency = number of unique order/customers
    #Monetary = Total spending / customer
    
    #my understanding is --> For recency you need time data, and for how often do they spend, like with the 
    #monetary value, you need to define in a particular time
    #so we can actually see the max and min years and stuffs to look at how much is a year gap and stuff more like a q1, q2, q3, q4
    
    
    #the code I wrote 
    '''df_operations = df_corrected['InvoiceDate']
    df_timeseries = df_operations.to_datetime()
    df_timeseries2 = df_timeseries.timestamp()
    print(df_timeseries2.max())
    print(df_timeseries2.min())'''
    
    #corrected code
    df_operations = df_corrected['InvoiceDate']
    
    #df_timeseries = df_operations.apply(pd.to_datetime) #my approach --> applies only for smaller datasets
    
    df_timeseries = pd.to_datetime(df_operations) #more professional way of doing it
    df_timeseries2 = df_timeseries.apply(lambda x : x.timestamp())
    print(f"Max timestamp: {df_timeseries2.max()}")
    print(f"Min timestamp: {df_timeseries2.min()}")
    
    customer_features = build_customer_intelligence(df_corrected)
    
    customer_features = add_temporal_features(customer_features, df_corrected)
    
    return customer_features

def add_temporal_features(customer_features, df_corrected):
    def determine_lifecycle_stage(row):
        if row['recency_days'] <= 30:
            return 'Active'
        elif row['recency_days'] <= 90:
            return 'Recent'
        elif row['recency_days'] <= 365:
            return 'Dormant'
        else:
            return 'Lost'
    
    customer_features['lifecycle_stage'] = customer_features.apply(determine_lifecycle_stage, axis=1)
    
    # Calculate purchase rhythm consistency
    customer_features['purchase_rhythm_score'] = 1 / (1 + customer_features['avg_days_between_orders'] / 30)
    customer_features['purchase_rhythm_score'] = customer_features['purchase_rhythm_score'].fillna(0)
    
    return customer_features
    

