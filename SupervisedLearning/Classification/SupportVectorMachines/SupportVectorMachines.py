import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import PowerTransformer, RobustScaler

# --- Function Definitions ---

def data_exploration(df_to_explore):
    """Prints a summary of the dataframe's structure and statistics."""
    info_buffer = io.StringIO()
    df_to_explore.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    print(f"----------- DataFrame Description -----------\n{df_to_explore.describe().to_string()}\n\n----------- DataFrame Shape -----------\n{df_to_explore.shape}\n\n----------- DataFrame Info -----------\n{info_output}")

def imputations(df_to_impute):
    """Handles missing values using IterativeImputer with a RandomForest estimator."""
    missing_values = df_to_impute.columns[df_to_impute.isnull().any()].tolist()
    rf_estimator = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)

    if not missing_values:
        print("\nThere are no missing values found.")
        return df_to_impute.copy()
    else:
        print("\nMissing values detected. Performing imputation...")
        imputer = IterativeImputer(max_iter=10, random_state=50, estimator=rf_estimator)
        # We fit and transform the data
        df_imputed_array = imputer.fit_transform(df_to_impute)
        # Create a new DataFrame with the imputed values
        df_transformed = pd.DataFrame(df_imputed_array, columns=df_to_impute.columns, index=df_to_impute.index)
        print("The values have been imputed.")
        return df_transformed

def automatic_outlier_analysis(df_to_analyze):
    """
    Uses OneClassSVM to detect outliers and separates the dataframe into inliers and outliers.
    """
    dfout = df_to_analyze.copy()
    
    # Initialize the OneClassSVM model. nu=0.1 means we expect about 10% of the data to be outliers.
    model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale', verbose=False)
    
    # Use the model to predict which rows are outliers. It returns 1 for inliers and -1 for outliers.
    # We exclude 'Id' from the model's consideration as it's just an identifier.
    dfout["outlier_score"] = model.fit_predict(dfout.drop('Id', axis=1))

    # Create two new dataframes based on the model's scores
    dfout_inliers = dfout[dfout["outlier_score"] == 1].copy()
    dfout_outliers = dfout[dfout["outlier_score"] == -1].copy()

    print(f"\nIdentified {len(dfout_outliers)} outliers out of {len(dfout)} total data points.")
    
    # Return all three dataframes for the next step
    return dfout, dfout_inliers, dfout_outliers

def automatic_outlier_handling(df_full, df_inliers, df_outliers, method='cap_quantile'):
    """
    Handles outliers using the results from the analysis step.
    The capping limits are derived from the 'df_inliers' dataframe, which was
    identified by the OneClassSVM model.
    """
    print(f"\n----------- Automatic Outlier Handling (Method: {method}) -----------")

    if method == 'remove':
        print(f"Original shape: {df_full.shape}")
        # Return only the inliers, dropping the score column
        df_handled = df_inliers.drop('outlier_score', axis=1)
        print(f"New shape after removing outliers: {df_handled.shape}")
        return df_handled

    # For capping methods, we start with a copy of the full dataframe
    df_handled = df_full.copy()
    
    # Iterate over each column to apply the capping
    for column in df_inliers.columns:
        # Skip columns that should not be capped
        if column in ['Id', 'quality', 'outlier_score']:
            continue

        lower_cap, upper_cap = 0, 0
        if method == 'cap_quantile':
            # This is the SVM-based method: limits are from the SVM-identified inliers
            lower_cap = df_inliers[column].quantile(0.01)
            upper_cap = df_inliers[column].quantile(0.99)
        elif method == 'cap_minmax':
            # This is also an SVM-based method, using min/max of the inliers
            lower_cap = df_inliers[column].min()
            upper_cap = df_inliers[column].max()
        else:
            raise ValueError("Invalid method. Choose 'cap_quantile', 'cap_minmax', or 'remove'.")
            
        # Apply the calculated caps to the data
        df_handled[column] = df_full[column].clip(lower=lower_cap, upper=upper_cap)

    # --- Verification Step ---
    if not df_outliers.empty and method in ['cap_quantile', 'cap_minmax']:
        # Get the first outlier for a before-and-after comparison
        outlier_index = df_outliers.index[0]
        print("\n----------- Verification -----------")
        print(f"Comparing original vs. handled values for outlier at index {outlier_index}:\n")
        comparison_df = pd.DataFrame({
            'Original Outlier': df_full.loc[outlier_index],
            'Handled (Capped)': df_handled.loc[outlier_index]
        })
        print(comparison_df.to_string())
    
    # Return the final, cleaned dataframe, removing the temporary score column
    return df_handled.drop('outlier_score', axis=1)


# --- Main Execution Pipeline ---

# 1. Load the initial dataset
dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()

# 2. Explore the data
data_exploration(df)

# 3. Handle any missing values
df_imputed = imputations(df)

# 4. Use SVM to find outliers
df_with_scores, df_inliers, df_outliers = automatic_outlier_analysis(df_imputed)

# 5. Handle the identified outliers (default method is robust capping)
df_final = automatic_outlier_handling(df_with_scores, df_inliers, df_outliers, method='cap_quantile')

# 6. Display the head of the final, processed dataframe
print("\n----------- Final Data Head -----------")
print(df_final.head().to_string())

    
