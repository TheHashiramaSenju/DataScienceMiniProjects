import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()
print(type(df))
print(df.index)


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
    
    '''
    estimators work in a good manner like adding a Machine learning model as a supervisor and 
    that machine learning model actually helps in making the data with more quality
    '''
    
    rf_estimator = RandomForestRegressor(n_estimators=10, #using small number of trees for speed
                                         max_depth=5,     #Limit tree depth
                                         random_state=42,
                                         n_jobs=-1) #use all CPU cores
    if not missing_values:#what else here can be like a alternative for random forest regressor?
        print("There are no values found")
        df_transformed = df.copy()
    else:
        imputer  = IterativeImputer(max_iter = 10, random_state=50, initial_strategy='mean', imputation_order='ascending', estimator=rf_estimator)
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
    zero_value_locs = []
    iqr_outlier_locs = []
    z_score_outlier_locs = []
    
    for columns_i in df_transformed.columns:
        
        q1, q3 = df_transformed[columns_i].quantile([.25, .75])
        IQR = q3 - q1
        
        lower_bound = q1 - 3.0 * IQR
        upper_bound = q3 - 3.0 * IQR
        
        mean = np.mean(df_transformed[columns_i])
        std_dev = np.std(df_transformed[columns_i])
        
        #this index stuff is more efficient because of this -> RangeIndex(start=0, stop=1143, step=1)
        for j in df_transformed.index:
            
            #the down value uses the range of index values and names of the columns from the for loop 
            value = df_transformed.at[j, columns_i]
            if value == 0:
                zero_value_locs.append((j, columns_i))
            if value > upper_bound or value < lower_bound:   
                iqr_outlier_locs.append((j, columns_i))
            if std_dev != 0:
                z_score  = (value - mean) / std_dev
                if abs(z_score) > 3:
                    z_score_outlier_locs.append(j, columns_i) 
    
    if zero_value_locs:
        print(f"\nDetected the outlier{len(set(iqr_outlier_locs))}zero")
    else:
        print("\nNo zero values have been detected")
    
    if iqr_outlier_locs:
        print(f"Detected {len(set(iqr_outlier_locs))} IQR based outlier")
    else:
        print("No IQR-based outliers detected")
        
    if z_score_outlier_locs:
        print(f"\nDetected {len(set(z_score_outlier_locs))} Z- score based outliers")
    else:
        print("No Z-Score based outliers detected")
    print("Outlier analysis had been complete")      
    
def manual_outlier_handling():
    pass
 
def automatic_outlier_analysis():
    
    #automatic outlier handling / model based outlier handling
    # we will look on to actually implementing the TF (neural network based) outlier analysis in the upcoming modules 
    
    dfout = df_transformed.copy()
    mldmodel = OneClassSVM(kernel='rbf',
                           degree=7,
                           gamma='scale',
                           coef0=0.0,
                           tol=0.001,
                           nu=0.5,
                           shrinking=True,
                           cache_size=200,
                           verbose=True,
                           max_iter=-1)
    
    dfout["outlier_score"] = mldmodel.fit_predict(dfout)
    #mistake i did
    #for col in dfout["outlier_score"]:
    #    if col == -1:
    #        print(f"we have an outlier here at {dfout.at[col, "outlier_score"]}") 
    
    #pandas are vectorized. So, dfout tends to take a whole row instead of a single value
    
    '''
    Vectorization means that when you apply an operation (like == 1) to a column, 
    pandas performs that operation on every single element of the column at once, 
    and returns a new column (a Series) containing all the results.

    '''      
    global dfout_filtered, dfout_nonfiltered
    dfout_filtered = dfout[dfout["outlier_score"] == 1].copy()
    dfout_nonfiltered = dfout[dfout["outlier_score"] == -1].copy()
    '''
    My Understanding : 
        Since, pandas are vectorized, which means it goes through the values and saves
        as a single row. Here we can pertty much deduce that
        df["outlier_score"] gives you the true/false values, and df[df["outlier_score] == 1]
        works in such a way that, the True/False values that are actually returned from the,
        df["outlier_score" == 1] is further more examined by the df[] outside and let's us 
        read an actual value or data at that place.
    
    The actual thing:
        P.S. Just a tiny typo to be aware of for your notes.

        The correct syntax for the comparison is to have the comparison operator
        outside the string that names the column.

        Correct:
        df['outlier_score'] == 1

        Incorrect:
        df['outlier_score' == 1]

        The second version incorrectly tries to compare the string 'outlier_score' to the number 1
        before trying to find a column with the result ('False').
        This is a small detail, but it's a very important one in Python!
    
        "Since pandas is vectorized, when we do a comparison like df['outlier_score'] == 1, it goes through all the values in that column and produces a new column (a Series) of True/False results.
        This True/False Series then acts as a filter. When we place it inside the outer df[] (like df[filter]), pandas uses it to select and return only the entire rows from the original df where the filter's value was True."
        You are absolutely right. That's exactly how it works. Your understanding of the process is spot on.
        (P.S. Just a tiny typo I noticed in your text—make sure the comparison is outside the quotes: df['outlier_score'] == 1, not df['outlier_score' == 1]. It's a small thing, but it makes a big difference in the code!)
    '''
    print(dfout_nonfiltered)
    return dfout_filtered #Do SVM interals

def automatic_outlier_handling(method = 'cap_quantile'):
    #we specifically use scalers and transformers here to make the data more reliable in the learning model
    #do the table comparison and the data type difference in datasets 
    
    """
    Handles outliers in the dataframe using one of several methods.

    Args:
        method (str): The method to use for outlier handling.
                      Options:
                      - 'cap_quantile' (default): Caps outliers at the 1st and 99th
                        percentiles of the inlier data.
                      - 'cap_minmax': Caps outliers at the absolute min and max
                        of the inlier data.
                      - 'remove': Removes outlier rows entirely.
    """
    
    if method == 'remove':
        #print(f"Original shape: {}")
        pass

    def data_modeling(X_train, y_train):
    svm_classifier = SVC()
    
    param_grid = {
        'C' : [0.1, 1, 10],
        'kernel' : ['linear', 'rbf', 'poly'], 
        'gamma'  : ['scale', 'auto'],
        'degree' : [2, 3, 4]
    }
    
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'Found the best parameters {best_params}')
    
    return best_params
    
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.svm import SVC

# Assume X_train, X_test, y_train, y_test are your RAW, unscaled data splits

# 1. Create a full pipeline with preprocessing and the classifier
model_pipeline = Pipeline(steps=[
    ('power_transform', PowerTransformer()),
    ('minmax_scaler', MinMaxScaler()),
    ('classifier', SVC(random_state=42)) # The model goes here
])

# 2. Define the parameter grid for the components inside the pipeline
#    Use the step name __ parameter name (e.g., 'classifier__C')
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__degree': [2, 3] # Only used by 'poly' kernel
}

# 3. Create the GridSearchCV object with the pipeline and parameter grid
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# 4. Fit the grid search on the ORIGINAL, UNPROCESSED training data
#    The pipeline handles the scaling correctly for each cross-validation fold
grid_search.fit(X_train, y_train)

# 5. Get the best parameters and the best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")



data_exploration()
imputations()
automatic_outlier_analysis()
#automatic_outlier_handling()


import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ... (all your existing code for preparing data and the pipeline)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# --- SAVE THE MODEL ---

# 1. Get the best pipeline from the grid search
best_pipeline = grid_search.best_estimator_

# 2. Define a filename
filename = 'wine_quality_model.joblib'

# 3. Save the pipeline to a file
joblib.dump(best_pipeline, filename)

print(f"Model saved to {filename}")


# predict.py

import joblib
import pandas as pd

# --- LOAD THE SAVED MODEL ---
print("Loading trained model...")
model = joblib.load('wine_quality_model.joblib')
print("Model loaded successfully.")

def predict_wine_quality(input_data):
    """
    Takes user input, prepares it, and returns a quality prediction.
    
    Args:
        input_data (dict): A dictionary where keys are feature names and 
                           values are the user's input.
                           
    Returns:
        int: The predicted wine quality.
    """
    
    # 1. The model was trained on a DataFrame with specific column names.
    #    You must create a DataFrame with the exact same column names.
    #    This is a critical step.
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # 2. Use the loaded pipeline to make a prediction.
    #    The pipeline will automatically handle the scaling and transformation.
    prediction = model.predict(input_df)
    
    # 3. The prediction is an array (e.g., [6]), so return the first item.
    return prediction[0]

if __name__ == '__main__':
    # --- EXAMPLE OF USER INPUT (like from a CLI or UI) ---
    # This simulates a user providing the characteristics of a wine.
    example_input = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
    
    # Get the prediction for the example input
    predicted_quality = predict_wine_quality(example_input)
    
    print("\n--- Prediction ---")
    print(f"Input Features: {example_input}")
    print(f"Predicted Wine Quality: {predicted_quality}")
Traceback (most recent call last):
  File "/mnt/data/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/UserInput.py", line 51, in <module>
    predict_wine_quality()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/mnt/data/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/UserInput.py", line 41, in predict_wine_quality
    data_transformed = transformer.transform(input_df)
                       ^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'transformer' where it is not associated with a value