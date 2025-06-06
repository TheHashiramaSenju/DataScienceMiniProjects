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

# Load dataset
df = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/WineQT.csv')
df_clean = df.copy()

# Feature analysis
print(df_clean.describe(), "\n")
print(df_clean.info(), "\n")
df_clean = df_clean[df_clean.columns].astype('float64')
print(df_clean.corr())

# Uncomment the block for Plotly visualizations
'''
df_clean = df_clean.select_dtypes(include=['float64', 'int64'])  # Select numerical columns
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
    height=300 * num_cols,  # Adjust height dynamically
    template="plotly_dark"
)
fig.show()
'''

# Data visualization
def visualization():
    print("Starting data visualization...")

    # Generate histograms, KDE plots, and scatterplots for each feature
    for i in df_clean.columns:
        plt.figure(figsize=(15, 15))
        sns.histplot(df_clean[i])
        plt.title(f'Histogram of {i}')
        plt.close() 

        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_clean[i], fill=True, bw_adjust=0.5)
        plt.title(f'KDE Plot of {i}')
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.scatterplot(y=df_clean[i], x=range(len(df_clean)))
        plt.title(f'Scatterplot of {i}')
        plt.close()

        global lower_bound, upper_bound
        q1, q3 = df_clean[i].quantile([.25, .75])
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR  

    # Heatmap of correlations
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.close()

    # Boxplot & Violin plot visualization
    plt.xticks(rotation=45, ha="right", fontsize=10) 
    sns.boxplot(data=df_clean)
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot.png', dpi=300)
    plt.close()

    plt.xticks(rotation=45, ha="right", fontsize=10) 
    sns.violinplot(data=df_clean)
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/violinplot.png', dpi=300)
    plt.close()

# Outlier detection
def outlier_detection():
    print("Analyzing outliers...")

    # Print data correlations and info
    print(df_clean.corr, "\n")
    print(df.info(), "\n")
    print(df.describe, "\n")

    # Identify missing values
    null_locations = df_clean[df_clean.isnull().any(axis=1)]
    if not null_locations.empty:
        print("Rows with missing values:\n", null_locations)
    else:
        print("No missing values detected.")

    # Detect zero values and outliers using IQR and Z-score
    z, w, v = [], [], []
    for i in df_clean.columns:
        for j in df_clean.index:
            if df_clean.at[j, i] == 0:
                z.append((j, i))
            elif df_clean.at[j, i] > upper_bound or df_clean.at[j, i] < lower_bound:
                w.append((j, i))
            mean = np.mean(df_clean[i])
            std_dev = np.std(df_clean[i])
            z_scores = (df_clean.at[j, i] - mean) / std_dev
            v.append(z_scores)
            if abs(z_scores) > 3:
                v.append((j, i))

# Outlier handling & scaling
def outlier_handling():
    print("Handling outliers...")

    # Initial boxplot to visualize raw outliers
    plt.figure(figsize=(15, 15))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_clean)
    plt.show()
    plt.close()

    # Outlier detection using Isolation Forest
    isolated_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
    df_clean["outlier_score"] = isolated_forest.fit_predict(df_clean)
    df_filtered = df_clean[df_clean["outlier_score"] == 1]

    # Apply Robust Scaling
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df_filtered)

    # Convert scaled data back into a DataFrame
    df_filtered = pd.DataFrame(df_scaled, columns=df_clean.columns)

    # Log transformation for skewed data
    df_filtered["Total sulfur dioxide"] = np.log1p(df_filtered["total sulfur dioxide"])

    # Boxplot after outlier handling
    plt.figure(figsize=(15, 15))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_filtered)
    plt.show(block=False)

    global df2
    df2 = df_filtered.copy()
    df_filtered["residual sugar"] = np.log1p(df_filtered["residual sugar"])
    df_filtered["chlorides"] = np.log1p(df_filtered["chlorides"])

    # Final visualization of cleaned dataset
    plt.figure(figsize=(15, 15))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_filtered)
    plt.show()

# Model implementation using Decision Trees
def implementation():
    print("Training Decision Tree model...")

    # Splitting the dataset
    X = df2.drop(columns='quality', axis=1)
    y = df2['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)

    # Visualizing the decision tree
    plt.figure(figsize=(15, 15))
    tree.plot_tree(model, filled=True)
    plt.show()

# Run processes
if __name__ == "__main__":
    visualization()       
    outlier_detection()
    outlier_handling()
    implementation()
