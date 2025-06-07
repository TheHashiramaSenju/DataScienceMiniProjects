import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn import tree 

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
    
    #we will fix outliers on the basis of extremities 1. Sulfur-Di-Oxide and total-sulfur dioxide 

    
    '''for index, values in enumerate(df_clean["total sulfur dioxide"]):
        if (values > upper_bound or values < lower_bound):
          df_clean.at[index, "total sulfur dioxide"] = df_clean["total sulfur dioxide"].mean()'''    
    '''Key-Take away, the IQR based cleaning above will lead the box to shrink because 
    the outlier datas are actually removed and replaced. But here in isolataaion forest we cap with 
    numbers for identification(we essentially are hiding it). IN robust scaler it removes data and ensures no
    spread actually occurs.'''
    
    plt.figure(figsize = (15, 15))
    plt.xticks(rotation=45, ha = "right", fontsize=10)
    sns.boxplot(data=df_clean)
    plt.show()
    plt.close()
    
    #1. Outlier handling -> data scaling -> data transformation
    #outlier 
    isolated_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
    df_clean["outlier_score"] = isolated_forest.fit_predict(df_clean)
    df_filtered = df_clean[df_clean["outlier_score"] == 1]
    
    #scaling
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df_filtered)   
    
    #scaled data which is actually an numpy array must be turned into a pandas dataframe  
    # Ensures that the restored DataFrame retains proper column names from the original dataset. 
    # Prevents column mismatch issues when assigning transformed data.
    df_filtered = pd.DataFrame(df_scaled, columns=df_clean.columns)
    
    #log transformation for highly skewed data
    df_filtered["Total sulfur dioxide"] = np.log1p(df_filtered["total sulfur dioxide"])
    
    plt.figure(figsize = (15, 15))
    plt.xticks(rotation=45, ha = "right", fontsize=10)
    sns.boxplot(data=df_filtered)
    plt.show(block = False)
    
    #df_filtered["residual sugar"], _ = boxcox(df_filtered["residual sugar"])
    
    '''
    Applies Box-Cox transformation to "column_name" Modifies "column_name" directly by replacing its 
    values The _ variable stores the lambda value, which determines the transformation power  Works only on positive values—data 
    must be strictly positive (greater than 0
    
    Important: Box-Cox only works for strictly positive data. If you have zeros or negatives, you must shift values before applying the transformation.
  
    The Box-Cox transformation is like magical shrinking and stretching dust you sprinkle on your toys. It does two things: 🔹 Shrinks the biggest toys so they’re not overwhelming. 🔹 Stretches the tiny toys so they don’t get ignored.

    After using the dust, your toy collection looks more balanced—kind of like how Box-Cox makes messy, uneven numbers become more uniform and easy to work with!
    '''
    
    #for index, value in df_filtered["residual sugar"].items():
    #    if (value < lower_bound or value > upper_bound):
    #        df_filtered.at[index, "residual sugar"] = df_filtered["residual sugar"].mode()
    #        print(f"Outlier found at index {index}, replacing with mean")
    
    global df2
    df2 = df_filtered.copy()
    df_filtered["residual sugar"] = np.log1p(df_filtered["residual sugar"])
    df_filtered["chlorides"] = np.log1p(df_filtered["chlorides"])
    

    plt.figure(figsize = (15, 15))
    plt.xticks(rotation=45, ha = "right", fontsize=10)
    sns.boxplot(data=df_filtered)
    plt.show()
    
    #the difference can be observed via theses plots


def implementation():
    #splitting of data
    X = df2.drop(columns='quality', axis=1)
    y = df2['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = DecisionTreeClassifier()
    
    #model instantiation
    model.fit(X_train, y_train)
    
    #model evaluvation, using trained model to make predictions on testing data 
    #using the predict function
    
    #splite more like X train and y train and x test and y test. So trains with X datas and 
    #comparing it with y test and later from learnt models we use x test and y test to check the accuracy of this 
    #prediction model
    
    '''EDA & Skew Check

    Histograms, KDEs, .skew(), QQ-plots.

    Transform Skew (log1p / Box-Cox / YJ) →

    Scale Features (Standard / MinMax / Robust / etc.) →

    Detect & Filter Outliers (IsolationForest / LOF / etc.) →

    Train Your Model (DecisionTree / RandomForest / SVM / …)'''


    
    y_pred = model.predict(X_test)
    #here on we can use evaluavtion metrics to better understand the data
    
    #model interpretation
    #used for analyzing the decision making progress by visualization
    
    plt.figure(figsize=(15, 15))
    tree.plot_tree(model, filled=True)
    
 
visualization()       
outlier_detection()
outlier_handling()
implementation()