import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib # Import joblib for saving/loading models
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer

# --- GLOBAL VARIABLES TO STORE FITTED OBJECTS ---
# This is usually done by returning them from functions or passing them around
# For simplicity in this script, we'll assign them globally right after fitting.
fitted_scaler = None
trained_model = None
fitted_imputer = None # If you add imputation


# Load dataset
df = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/WineQT.csv')
df_clean = df.copy()

# Feature analysis (unchanged for this part)
print(df_clean.describe(), "\n")
print(df_clean.info(), "\n")
df_clean = df_clean[df_clean.columns].astype('float64')
print(df_clean.corr())

# Data visualization (unchanged - plots are saved)
def visualization():
    print("Starting data visualization...")
    global lower_bound, upper_bound # Define these globally as they are used in outlier_detection
    for i in df_clean.columns:
        # Save plots to avoid showing them during script run in background
        plt.figure(figsize=(6, 4))
        sns.histplot(df_clean[i])
        plt.title(f'Histogram of {i}')
        plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/hist_{i}.png', dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_clean[i], fill=True, bw_adjust=0.5)
        plt.title(f'KDE Plot of {i}')
        plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/kde_{i}.png', dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.scatterplot(y=df_clean[i], x=range(len(df_clean)))
        plt.title(f'Scatterplot of {i}')
        plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/scatter_{i}.png', dpi=150)
        plt.close()

        q1, q3 = df_clean[i].quantile([.25, .75])
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR

    plt.figure(figsize=(20, 20))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/heatmap.png', dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10)) # Adjusted size for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_clean)
    plt.title("Initial Boxplot of Features")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_initial.png', dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10)) # Adjusted size
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.violinplot(data=df_clean)
    plt.title("Initial Violin Plot of Features")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/violinplot_initial.png', dpi=300)
    plt.close()
    print("Data visualization complete.")

# Missing Value Imputation Function (New addition)
def missing_value_imputation(df_input):
    print("Checking for and handling missing values with Iterative Imputer...")
    global fitted_imputer # To store the fitted imputer

    # Identify columns with missing values
    cols_with_missing = df_input.columns[df_input.isnull().any()].tolist()

    if not cols_with_missing:
        print("No missing values detected. Skipping imputation.")
        return df_input.copy()

    print(f"Columns with missing values: {cols_with_missing}")

    # Initialize and fit the IterativeImputer
    # We choose BayesianRidge as it's generally robust, but you can experiment
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='mean', estimator=None) # estimator=None uses BayesianRidge
    fitted_imputer = imputer.fit(df_input) # Fit the imputer on the full dataset (or training set)
    df_imputed_array = fitted_imputer.transform(df_input)

    df_imputed = pd.DataFrame(df_imputed_array, columns=df_input.columns, index=df_input.index)

    print("Missing values handled.")
    print("Missing values after imputation:\n", df_imputed.isnull().sum())
    return df_imputed


# Outlier detection (unchanged logic, but output modified to use global bounds)
def outlier_detection():
    print("Analyzing outliers...")

    # Identify missing values (now handled by missing_value_imputation)
    # This block is now mostly for informational purposes, as NaN should be handled
    null_locations = df_clean[df_clean.isnull().any(axis=1)]
    if not null_locations.empty:
        print("Rows with missing values before imputation:\n", null_locations)
    else:
        print("No missing values detected before imputation.")


    # Detect zero values and outliers using IQR and Z-score (informational, not removal)
    zero_values_locs = []
    iqr_outliers_locs = []
    z_score_outliers_locs = []

    for col_name in df_clean.columns:
        # Assuming lower_bound and upper_bound were calculated in visualization() and are global
        q1, q3 = df_clean[col_name].quantile([.25, .75])
        IQR = q3 - q1
        col_lower_bound = q1 - 1.5 * IQR
        col_upper_bound = q3 + 1.5 * IQR

        mean = np.mean(df_clean[col_name])
        std_dev = np.std(df_clean[col_name])

        for j in df_clean.index:
            value = df_clean.at[j, col_name]
            if value == 0:
                zero_values_locs.append((j, col_name))
            if value > col_upper_bound or value < col_lower_bound:
                iqr_outliers_locs.append((j, col_name))
            if std_dev != 0: # Avoid division by zero
                z_score = (value - mean) / std_dev
                if abs(z_score) > 3:
                    z_score_outliers_locs.append((j, col_name))

    if zero_values_locs:
        print(f"\nDetected {len(zero_values_locs)} zero values at: {set(zero_values_locs)}")
    else:
        print("\nNo zero values detected.")

    if iqr_outliers_locs:
        print(f"Detected {len(iqr_outliers_locs)} IQR-based outliers at: {set(iqr_outliers_locs)}")
    else:
        print("No IQR-based outliers detected.")

    if z_score_outliers_locs:
        print(f"Detected {len(z_score_outliers_locs)} Z-score-based outliers at: {set(z_score_outliers_locs)}")
    else:
        print("No Z-score-based outliers detected.")


# Outlier handling & scaling
def outlier_handling(df_input):
    print("Handling outliers and scaling data...")
    global fitted_scaler, df2 # To store the fitted scaler and the processed dataframe

    # Initial boxplot to visualize raw outliers (now uses df_input)
    plt.figure(figsize=(15, 10))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_input)
    plt.title("Boxplot Before Isolation Forest Filtering and Scaling")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_pre_isoforest.png', dpi=300)
    plt.close()

    # Outlier detection using Isolation Forest
    isolated_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
    # Fit and predict on the *input* DataFrame
    df_input["outlier_score"] = isolated_forest.fit_predict(df_input)
    df_filtered = df_input[df_input["outlier_score"] == 1].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Apply Robust Scaling
    scaler = RobustScaler()
    # Fit the scaler on the *filtered* data and then transform it
    # We fit only on features, excluding 'quality' and 'outlier_score'
    features_to_scale = df_filtered.drop(columns=["quality", "outlier_score"], errors='ignore').columns
    fitted_scaler = scaler.fit(df_filtered[features_to_scale]) # Store the fitted scaler
    df_scaled_array = fitted_scaler.transform(df_filtered[features_to_scale])

    # Convert scaled data back into a DataFrame, keeping 'quality' and 'outlier_score'
    df_filtered_scaled = pd.DataFrame(df_scaled_array, columns=features_to_scale, index=df_filtered.index)
    # Add back 'quality' and 'outlier_score'
    df_filtered_scaled["quality"] = df_filtered["quality"]
    df_filtered_scaled["outlier_score"] = df_filtered["outlier_score"]


    # Log transformation for skewed data (only for features that might be skewed)
    # Apply to the *scaled* features where appropriate
    # NOTE: You should have already checked for skewness during visualization
    # We apply log1p *after* scaling, if the scaled data still exhibits skewness.
    # It's more common to do transformations *before* scaling if the transformation itself
    # is meant to normalize the distribution for the benefit of the scaler or the model.
    # Let's adjust this for clarity: transformations usually happen *before* scaling for numerical features.
    # However, if features were scaled to a robust range (e.g. median 0, IQR 1) and then log-transformed,
    # the interpretation of log1p might change.
    # Given your original code structure, let's keep it here but note best practice:
    # Transformations like boxcox/log should ideally precede scaling unless the goal is to make the scaled features more normal.

    # Re-apply log transformations *after* scaling, if necessary.
    # Ensure these columns exist in df_filtered_scaled.
    # We are applying them on the *transformed* scaled data.
    # IMPORTANT: The model will expect these transformations on *new* data too.
    if "total sulfur dioxide" in df_filtered_scaled.columns:
        df_filtered_scaled["total sulfur dioxide"] = np.log1p(df_filtered_scaled["total sulfur dioxide"])
    if "residual sugar" in df_filtered_scaled.columns:
        df_filtered_scaled["residual sugar"] = np.log1p(df_filtered_scaled["residual sugar"])
    if "chlorides" in df_filtered_scaled.columns:
        df_filtered_scaled["chlorides"] = np.log1p(df_filtered_scaled["chlorides"])

    df2 = df_filtered_scaled.copy() # Store the fully processed dataframe for model training

    # Final visualization of cleaned dataset
    plt.figure(figsize=(15, 10))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df2.drop(columns=["quality", "outlier_score"], errors='ignore')) # Exclude target/score for boxplot
    plt.title("Boxplot After Outlier Handling, Scaling, and Transformations")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_final.png', dpi=300)
    plt.close()
    print("Outlier handling and scaling complete.")
    return df2 # Return the processed DataFrame

# Model implementation using Decision Trees
def implementation(df_processed):
    print("Training Decision Tree model...")
    global trained_model # To store the trained model

    X = df_processed.drop(columns=["quality", "outlier_score"], errors='ignore') # Exclude outlier_score from features
    y = df_processed["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added test_size for clarity
    model = DecisionTreeClassifier(random_state=42)
    trained_model = model.fit(X_train, y_train) # Store the trained model

    # 1) Plot the full‐data decision tree
    plt.figure(figsize=(20,15)) # Increased size for better readability
    tree.plot_tree(trained_model, feature_names=X.columns, class_names=[str(c) for c in trained_model.classes_], filled=True, fontsize=8)
    plt.title("Full Decision Tree")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/full_dt.png', dpi=300)
    plt.close()

    # 2) Now train & plot on a small sample if you want a “zoomed‐in” tree
    sampleData = df_processed.sample(50, random_state=42) # pick 50 random rows
    X_sample = sampleData.drop(columns=["quality", "outlier_score"], errors='ignore')
    y_sample = sampleData["quality"]
    sampleModel = DecisionTreeClassifier(random_state=42)
    sampleModel.fit(X_sample, y_sample)

    plt.figure(figsize=(12,12)) # Adjusted size
    tree.plot_tree(
      sampleModel,
      feature_names = X_sample.columns,
      class_names   = [str(c) for c in sampleModel.classes_],
      filled        = True,
      fontsize      = 10
    )
    plt.title("Decision Tree on 50-row Sample")
    plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/sample_dt.png', dpi=300)
    plt.close()
    print("Decision Tree training complete. Plots saved.")

    # Model Evaluation (Crucial!)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = trained_model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Model evaluation complete.")


# Run processes and save artifacts
if __name__ == "__main__":
    print("--- Starting Training Pipeline ---")
    visualization()
    # Ensure df_clean is available globally or passed
    df_clean = missing_value_imputation(df_clean) # Impute missing values first

    # Update df_clean for outlier_detection to reflect imputed values
    # Note: The outlier_detection function as written is more for reporting,
    # and doesn't modify df_clean. The actual filtering happens in outlier_handling.
    outlier_detection() # This function still uses df_clean (global) for info

    df_processed_for_model = outlier_handling(df_clean) # Pass the imputed df
    implementation(df_processed_for_model)

    # --- SAVE FITTED OBJECTS ---
    print("\n--- Saving Fitted Objects ---")
    try:
        joblib.dump(trained_model, 'decision_tree_model.joblib')
        print("Decision Tree model saved as 'decision_tree_model.joblib'")

        # Ensure scaler is fitted and available globally.
        # It's better to pass it or return it from `outlier_handling`.
        if fitted_scaler:
            joblib.dump(fitted_scaler, 'robust_scaler.joblib')
            print("Robust Scaler saved as 'robust_scaler.joblib'")
        else:
            print("Robust Scaler not fitted or not assigned to `fitted_scaler`. Cannot save.")

        if fitted_imputer:
            joblib.dump(fitted_imputer, 'iterative_imputer.joblib')
            print("Iterative Imputer saved as 'iterative_imputer.joblib'")
        else:
            print("Iterative Imputer not fitted or not assigned to `fitted_imputer`. Cannot save.")

        # Save feature names as well, essential for consistent prediction
        feature_names = df_processed_for_model.drop(columns=["quality", "outlier_score"], errors='ignore').columns.tolist()
        joblib.dump(feature_names, 'feature_names.joblib')
        print("Feature names saved as 'feature_names.joblib'")

    except Exception as e:
        print(f"Error saving objects: {e}")

    print("--- Training Pipeline Complete ---")