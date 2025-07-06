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
from sklearn.model_selection import GridSearchCV
import joblib


dataset = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/SupportVectorMachines/WineQT.csv')
df = dataset.copy()

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
    
    featuers_for_model = dfout.drop('quality', axis=1)
    outlier_predictions = model.fit_predict(featuers_for_model)
    dfout['outlier_score'] = outlier_predictions

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
    print(df_handled.drop('outlier_score', axis=1))
    return df_handled.drop('outlier_score', axis=1)

def feature_scaling_and_transformations(df_processed):
    
    transformer = PowerTransformer()
    scaler = MinMaxScaler()
    full_data = df_processed.drop('quality', axis=1)
    train_target = df_processed['quality']
    X_train, X_test, y_train, y_test  = train_test_split(full_data, train_target, test_size = 0.2, random_state = 44 )
    
    X_train_transformed  = transformer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    
    X_test_transformed = transformer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_transformed)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, transformer, scaler
    
    
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
    
    return grid_search

def model_training (X_train, X_test, y_train, y_test, best_params):
    svm_classifier_optimized = SVC(**best_params)
    svm_classifier_optimized.fit(X_train, y_train)
    
    score = svm_classifier_optimized.score(X_test, y_test)
    print(f'The accuracy score of optimized SVM is {score}')
    

if __name__ == '__main__':
    
    data_exploration(df)
    df_imputed = imputations(df)
    df_with_scores, df_inliers, df_outliers = automatic_outlier_analysis(df_imputed)
    df_final = automatic_outlier_handling(df_with_scores, df_inliers, df_outliers)
    
    X_train_processed, X_test_processed, y_train, y_test, fitted_transformer, fitted_scaler = feature_scaling_and_transformations(df_final)
    grid_search_result = data_modeling(X_train_processed, y_train)
    
    model_training(X_train_processed, X_test_processed, y_train, y_test, grid_search_result.best_params_)
    
    
    print("Model saving")
    joblib.dump(fitted_transformer, 'wine_transformer.joblib')
    joblib.dump(fitted_scaler, 'wine_scaler.joblib')
    joblib.dump(grid_search_result.best_estimator_, 'wine_svm_model.joblib')
    print("Components have been saved successfully")
    




