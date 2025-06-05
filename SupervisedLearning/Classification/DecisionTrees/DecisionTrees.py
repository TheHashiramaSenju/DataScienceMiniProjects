import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.subplots as sp

df = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/WineQT.csv')
df_clean = df.copy()

#Feature analysis 
print(df_clean.describe())
print()
print(df_clean.info())
print()

df_clean = df_clean[df_clean.columns].astype('float64')
print(df_clean.corr)

#uncomment the block for plotly (believe me it is pretty amazing)
'''df_clean = df_clean.select_dtypes(include=['float64', 'int64'])  # Select only numerical columns
num_cols = len(df_clean.columns)
# Create subplots dynamically
fig = sp.make_subplots(rows=num_cols, cols=1, shared_xaxes=False, subplot_titles=df_clean.columns)
# Add histogram traces
for idx, col in enumerate(df_clean.columns):
    fig.add_trace(
        go.Histogram(x=df_clean[col], nbinsx=30, name=col, opacity=0.7),
        row=idx + 1, col=1
    )
# Layout customization
fig.update_layout(
    title_text="Interactive Frequency Distribution",
    showlegend=False,
    height=300 * num_cols,  # Dynamically adjust height
    template="plotly_dark"  # Dark theme for aesthetics
)
# Show the interactive plot
fig.show()'''
#the snippet ends here

#Data-Visualization 
def visualization():
    print("The visualization process have been started")

    for i in df_clean.columns:
        plt.figure(figsize=(15, 15))  # Create a new figure for each plot
              
        # Histogram
        sns.histplot(df_clean[i])
        plt.title(f'Histogram of {i}')
        #plt.savefig(f'{i}_histogram.png', dpi=300)  # Save as PNG
        plt.close()  # Close figure to avoid overlap
        
        # KDE plot
        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_clean[i], fill=True, bw_adjust=0.5)
        plt.title(f'KDE Plot of {i}')
        #plt.savefig(f'{i}_kde.png', dpi=300)
        plt.close()
        
        # Scatterplot (if applicable, might not be meaningful for 1D data)
        plt.figure(figsize=(6, 4))
        sns.scatterplot(y=df_clean[i], x=range(len(df_clean)))  # Scatter visualization
        plt.title(f'Scatterplot of {i}')
        #plt.savefig(f'{i}_scatter.png', dpi=300)
        plt.close()
        
        global lower_bound, upper_bound
        a = []
        q1, q3 = df_clean[i].quantile([.25, .75])
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        a.append((lower_bound, upper_bound))    
    
    
    # **Heatmap of correlation**
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    #plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()
    #heatmap of correlation
    sns.heatmap(df_clean.corr(), annot=True)
    plt.close()
    plt.xticks(rotation=45, ha="right",fontsize=10 ) 
    sns.boxplot(data=df_clean)
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot.png', dpi=300)
    plt.close()
    plt.xticks(rotation=45, ha="right",fontsize=10 ) 
    sns.violinplot(data=df_clean)
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/violinplot.png', dpi=300)
    plt.close()

def outlier_detection():
    #from the boxplot it is pretty evident about the outliers
    #common outlier detections and later we will do the column specific outlier-handling
    
    
    #gathering intel on outliers and table information 
    print("Data Intel")
    print(df_clean.corr, end="\n")
    print(df.info(), end = "\n")
    print(df.describe, end = "\n")
    
    null_locations = df_clean[df_clean.isnull().any(axis=1)]
    if not null_locations.empty:
        print("Rows with missing values are :")
        print(null_locations)
    else:
        print("No missing values")
    
    z,w,v = [], [], []
    for i in df_clean.columns:
        for j in df_clean.index:
            if (df_clean.at[j, i] == 0):
                print(f"Zero values found at {(j, i)}")
                z.append((j, i))
            elif (df_clean.at[j, i] > upper_bound or df_clean.at[j,i] < lower_bound):
                #print(f"IQR outlier values have been found here{df_clean.at[j, i]}")
                w.append((j,i))
            else:
                pass
            mean = np.mean(df_clean[i])
            std_dev = np.std(df_clean[i])
            z_scores = (df_clean.at[j, i] - mean) / std_dev
            v.append(z_scores)
            if abs(z_scores) > 3:
                v.append((j, i))
                
def outlier_handling():
    #we got several 0 values at citric acid, but most of the values of citric acid are pretty close to 0, so imputing them with another value would be not good 
    #we will try other methods in case severe outliers happened 
    #it is pretty evident that some columns have extreme outliers
    
    #we will fix outliers on the basis of extremities 
    
            
visualization()       
outlier_detection()