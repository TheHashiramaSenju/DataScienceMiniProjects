import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


def data_exploration():
    dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
    df = dataset.copy()
    
    print(df.corr())
    print(df.head())

data_exploration()