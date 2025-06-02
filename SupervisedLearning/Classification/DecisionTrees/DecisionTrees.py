import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.subplots as sp

df = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/WineQT.csv')
df_clean = df.copy()

print(df_clean.describe())
print()
print(df_clean.info())
print()

df_clean = df_clean[df_clean.columns].astype('float64')
print(df_clean.isnull().sum())
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
        plt.figure(figsize=(6, 4))  # Create a new figure for each plot
        
        # Histogram
        sns.histplot(df_clean[i])
        plt.title(f'Histogram of {i}')
        plt.savefig(f'{i}_histogram.png', dpi=300)  # Save as PNG
        plt.close()  # Close figure to avoid overlap
        
        # KDE plot
        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_clean[i], fill=True, bw_adjust=0.5)
        plt.title(f'KDE Plot of {i}')
        plt.savefig(f'{i}_kde.png', dpi=300)
        plt.close()
        
        # Scatterplot (if applicable, might not be meaningful for 1D data)
        plt.figure(figsize=(6, 4))
        sns.scatterplot(y=df_clean[i], x=range(len(df_clean)))  # Scatter visualization
        plt.title(f'Scatterplot of {i}')
        plt.savefig(f'{i}_scatter.png', dpi=300)
        plt.close()
    
    # **Heatmap of correlation**
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()
    
        
    #heatmap of correlation
    sns.heatmap(df_clean.corr(), annot=True)





visualization()