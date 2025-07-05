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

def data_exploration(df_to_explore):
    info_buffer = io.StringIO()
    df_to_explore.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    print(f"----------- DataFrame Description -----------\n{df_to_explore.describe().to_string()}\n\n----------- DataFrame Shape -----------\n{df_to_explore.shape}\n\n----------- DataFrame Info -----------\n{info_output}")

def imputations(df_to_impute):
    missing_values = df_to_impute.columns[df_to_impute.isnull().any()].tolist()
    rf_estimator = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)

    if not missing_values:
        print("\nThere are no missing values found.")
        return df_to_impute.copy()
    else:
        print("\nMissing values detected. Performing imputation...")
        imputer = IterativeImputer(max_iter=10, random_state=50, estimator=rf_estimator)
        df_imputed_array = imputer.fit_transform(df_to_impute)
        df_transformed = pd.DataFrame(df_imputed_array, columns=df_to_impute.columns, index=df_to_impute.index)
        print("The values have been imputed.")
        return df_transformed

def automatic_outlier_analysis(df_to_analyze):
    dfout = df_to_analyze.copy()
    
    model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale', verbose=True)
    
    dfout["outlier_score"] = model.fit_predict(dfout.drop('Id', axis=1))

    dfout_inliers = dfout[dfout["outlier_score"] == 1].copy()
    dfout_outliers = dfout[dfout["outlier_score"] == -1].copy()

    print(f"\nIdentified {len(dfout_outliers)} outliers out of {len(dfout)} total data points.")
    
    return dfout, dfout_inliers, dfout_outliers

def automatic_outlier_handling(df_full, df_inliers, df_outliers, method='cap_quantile'):
    print(f"\n----------- Automatic Outlier Handling (Method: {method}) -----------")

    if method == 'remove':
        print(f"Original shape: {df_full.shape}")
        df_handled = df_inliers.drop('outlier_score', axis=1)
        print(f"New shape after removing outliers: {df_handled.shape}")
        return df_handled

    df_handled = df_full.copy()
    
    for column in df_inliers.columns:
        if column in ['Id', 'quality', 'outlier_score']:
            continue

        lower_cap, upper_cap = 0, 0
        if method == 'cap_quantile':
            lower_cap = df_inliers[column].quantile(0.01)
            upper_cap = df_inliers[column].quantile(0.99)
        elif method == 'cap_minmax':
            lower_cap = df_inliers[column].min()
            upper_cap = df_inliers[column].max()
        else:
            raise ValueError("Invalid method. Choose 'cap_quantile', 'cap_minmax', or 'remove'.")
            
        df_handled[column] = df_full[column].clip(lower=lower_cap, upper=upper_cap)

    if not df_outliers.empty and method in ['cap_quantile', 'cap_minmax']:
        outlier_index = df_outliers.index[0]
        print("\n----------- Verification -----------")
        print(f"Comparing original vs. handled values for outlier at index {outlier_index}:\n")
        comparison_df = pd.DataFrame({
            'Original Outlier': df_full.loc[outlier_index],
            'Handled (Capped)': df_handled.loc[outlier_index]
        })
        print(comparison_df.to_string())
    
    return df_handled.drop('outlier_score', axis=1)


dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()

data_exploration(df)

df_imputed = imputations(df)

df_with_scores, df_inliers, df_outliers = automatic_outlier_analysis(df_imputed)

df_final = automatic_outlier_handling(df_with_scores, df_inliers, df_outliers, method='cap_quantile')

print("\n----------- Final Data Head -----------")
print(df_final.head().to_string())
