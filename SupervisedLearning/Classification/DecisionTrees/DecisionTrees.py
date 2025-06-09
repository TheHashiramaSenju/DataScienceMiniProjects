import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import joblib # Used for saving/loading trained models and preprocessors
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer

# --- Global Variables for Fitted Objects ---
# These variables will store the trained model and fitted preprocessors
# to be saved for later use in prediction.
fitted_scaler = None
trained_model = None
fitted_imputer = None
lower_bound = None
upper_bound = None


# --- 1. Data Loading and Initial Inspection ---
# Load the dataset
df = pd.read_csv('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/WineQT.csv')
df_clean = df.copy() # Create a working copy of the DataFrame

# Display basic data information and statistics
print("--- Initial Data Analysis ---")
print("DataFrame Description:\n", df_clean.describe(), "\n")
print("DataFrame Info:\n", df_clean.info(), "\n")

# Ensure all columns are numeric (float64) for consistent processing
df_clean = df_clean[df_clean.columns].astype('float64')
print("DataFrame Correlation Matrix:\n", df_clean.corr())


# --- 2. Data Visualization ---
def visualize_data(df_input):
    """
    Generates and saves various plots for data distribution and relationships.
    Calculates initial IQR bounds used for outlier detection.
    """
    print("\n--- Starting Data Visualization ---")
    global lower_bound, upper_bound # Used for IQR bounds in outlier_detection

    for col in df_input.columns:
        # Generate and save Histograms
        plt.figure(figsize=(6, 4))
        sns.histplot(df_input[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        #plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/hist_{col}.png', dpi=150)
        plt.close()

        # Generate and save Scatterplots (vs. index)
        plt.figure(figsize=(6, 4))
        sns.scatterplot(y=df_input[col], x=range(len(df_input)))
        plt.title(f'Scatterplot of {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.tight_layout()
        #plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/scatter_{col}.png', dpi=150)
        plt.close()

        # Calculate IQR for each column (used for initial outlier detection)
        q1, q3 = df_input[col].quantile([.25, .75])
        IQR = q3 - q1
        # Store bounds for the last column processed, or handle per-column
        # For this context, assuming these are illustrative and not directly used for filtering in next step
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR

    # Generate and save Correlation Heatmap
    plt.figure(figsize=(12, 10)) # Adjusted size for better display
    sns.heatmap(df_input.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/heatmap.png', dpi=300)
    plt.close()

    # Generate and save Boxplot of all features
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_input)
    plt.title("Initial Boxplot of Features (Before Preprocessing)")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_initial.png', dpi=300)
    plt.close()

    # Generate and save Violin plot of all features
    plt.figure(figsize=(15, 8))
    sns.violinplot(data=df_input)
    plt.title("Initial Violin Plot of Features (Before Preprocessing)")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/violinplot_initial.png', dpi=300)
    plt.close()
    print("Data visualization complete. Plots saved to 'Plots' directory.")


# --- 3. Missing Value Imputation ---
def impute_missing_values(df_input):
    """
    Handles missing values using IterativeImputer.
    Fits the imputer and stores it globally for consistent preprocessing.
    """
    print("\n--- Handling Missing Values ---")
    global fitted_imputer

    # Identify columns with missing values
    cols_with_missing = df_input.columns[df_input.isnull().any()].tolist()

    if not cols_with_missing:
        print("No missing values detected. Skipping imputation.")
        return df_input.copy()

    print(f"Columns with missing values: {cols_with_missing}")

    # Initialize and fit IterativeImputer
    # BayesianRidge estimator is used by default if estimator=None
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='mean')
    fitted_imputer = imputer.fit(df_input) # Fit imputer on the input DataFrame
    df_imputed_array = fitted_imputer.transform(df_input)

    df_imputed = pd.DataFrame(df_imputed_array, columns=df_input.columns, index=df_input.index)

    print("Missing values imputation complete.")
    print("Missing values after imputation:\n", df_imputed.isnull().sum())
    return df_imputed


# --- 4. Outlier Analysis (Informational) ---
def analyze_outliers(df_input):
    """
    Analyzes and reports zero values, IQR-based, and Z-score-based outliers.
    This function is for reporting and does not remove outliers.
    """
    print("\n--- Analyzing Outliers (Informational) ---")

    # Check for missing values (should be handled by previous step)
    if df_input.isnull().any().any():
        print("Warning: Missing values still present. Review imputation step.")
    else:
        print("No missing values detected after imputation.")

    zero_values_locs = []
    iqr_outliers_locs = []
    z_score_outliers_locs = []

    for col_name in df_input.columns:
        # Calculate IQR and Z-score for each column
        q1, q3 = df_input[col_name].quantile([.25, .75])
        IQR = q3 - q1
        col_lower_bound = q1 - 1.5 * IQR
        col_upper_bound = q3 + 1.5 * IQR

        mean = np.mean(df_input[col_name])
        std_dev = np.std(df_input[col_name])

        for j in df_input.index:
            value = df_input.at[j, col_name]
            if value == 0:
                zero_values_locs.append((j, col_name))
            if value > col_upper_bound or value < col_lower_bound:
                iqr_outliers_locs.append((j, col_name))
            if std_dev != 0: # Avoid division by zero
                z_score = (value - mean) / std_dev
                if abs(z_score) > 3:
                    z_score_outliers_locs.append((j, col_name))

    if zero_values_locs:
        print(f"\nDetected {len(set(zero_values_locs))} zero values.")
    else:
        print("\nNo zero values detected.")

    if iqr_outliers_locs:
        print(f"Detected {len(set(iqr_outliers_locs))} IQR-based outliers.")
    else:
        print("No IQR-based outliers detected.")

    if z_score_outliers_locs:
        print(f"Detected {len(set(z_score_outliers_locs))} Z-score-based outliers.")
    else:
        print("No Z-score-based outliers detected.")
    print("Outlier analysis complete.")


#5. Outlier Handling and Feature Scaling
def handle_outliers_and_scale(df_input):
    """
    Filters outliers using Isolation Forest, applies Robust Scaling,
    and performs log transformations on specified features.
    The fitted scaler is stored globally.
    """
    print("\n--- Handling Outliers and Scaling Data ---")
    global fitted_scaler

    # Initial boxplot before filtering
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_input.drop(columns='quality', errors='ignore')) # Exclude target for visualization
    plt.title("Boxplot Before Isolation Forest Filtering and Scaling")
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_pre_isoforest.png', dpi=300)
    plt.close()

    # Outlier detection and filtering using Isolation Forest
    isolated_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
    df_input["outlier_score"] = isolated_forest.fit_predict(df_input)
    df_filtered = df_input[df_input["outlier_score"] == 1].copy() # Keep inliers only (a rather interesting one)  
    
    
    # Identify features for scaling (exclude target and outlier_score)
    features_to_scale = df_filtered.drop(columns=["quality", "outlier_score"], errors='ignore').columns
    

    # Apply Robust Scaling - fit on filtered data, transform it
    scaler = RobustScaler()
    fitted_scaler = scaler.fit(df_filtered[features_to_scale]) # Fit and store the scaler
    df_scaled_array = fitted_scaler.transform(df_filtered[features_to_scale])
      

    # Convert scaled features back to DataFrame
    df_processed = pd.DataFrame(df_scaled_array, columns=features_to_scale, index=df_filtered.index)
    # Re-add 'quality' and 'outlier_score' to the processed DataFrame
    df_processed["quality"] = df_filtered["quality"]
    df_processed["outlier_score"] = df_filtered["outlier_score"]
 
    # Apply log1p transformations for potentially skewed features (post-scaling)
    # These transformations should be consistent with training data when making predictions.
    for col in ["total sulfur dioxide", "residual sugar", "chlorides"]:
        if col in df_processed.columns:
            df_processed[col] = np.log1p(df_processed[col])

    # Final boxplot of processed data
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_processed.drop(columns=["quality", "outlier_score"], errors='ignore'))
    plt.title("Boxplot After Outlier Handling, Scaling, and Transformations")
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_final.png', dpi=300)
    plt.close()

    print("Outlier handling and scaling complete.")
    return df_processed # Return the fully processed DataFrame


# --- 6. Model Training and Evaluation ---
def train_and_evaluate_model(df_processed):
    """
    Trains a Decision Tree Classifier, evaluates its performance,
    and plots the decision trees. The trained model is stored globally.
    """
    print("\n--- Training Decision Tree Model ---")
    global trained_model

    # Define features (X) and target (y)
    X = df_processed.drop(columns=["quality", "outlier_score"], errors='ignore')
    y = df_processed["quality"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    trained_model = model.fit(X_train, y_train) # Fit and store the trained model

    # Plot the full decision tree
    plt.figure(figsize=(20, 15))
    tree.plot_tree(trained_model, feature_names=X.columns,
                   class_names=[str(c) for c in trained_model.classes_],
                   filled=True, fontsize=8)
    plt.title("Full Decision Tree")
    plt.tight_layout()
    #plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/full_dt.png', dpi=300)
    plt.close()

    # Plot a smaller sample decision tree for better readability
    sample_df = df_processed.sample(50, random_state=42)
    X_sample = sample_df.drop(columns=["quality", "outlier_score"], errors='ignore')
    y_sample = sample_df["quality"]
    sample_model = DecisionTreeClassifier(random_state=42)
    sample_model.fit(X_sample, y_sample)

    plt.figure(figsize=(12, 12))
    tree.plot_tree(sample_model, feature_names=X_sample.columns,
                   class_names=[str(c) for c in sample_model.classes_],
                   filled=True, fontsize=10)
    plt.title("Decision Tree on 50-row Sample")
    plt.tight_layout()
    ##plt.savefig('/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/sample_dt.png', dpi=300)
    plt.close()
    print("Decision Tree training complete. Tree plots saved.")

    # Evaluate the model on the test set
    print("\n--- Model Evaluation ---")
    y_pred = trained_model.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Model evaluation complete.")

    # Return feature names, essential for consistent prediction
    return X.columns.tolist()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Starting Machine Learning Training Pipeline ---")

    # Run data visualization
    visualize_data(df_clean)

    # Handle missing values
    df_clean = impute_missing_values(df_clean)

    # Analyze outliers (informational)
    analyze_outliers(df_clean)

    # Handle outliers, scale features, and apply transformations
    df_processed_for_model = handle_outliers_and_scale(df_clean)

    # Train and evaluate the Decision Tree model
    feature_names = train_and_evaluate_model(df_processed_for_model)

    # --- Save Fitted Objects ---
    print("\n--- Saving Fitted Objects ---")
    try:
        joblib.dump(trained_model, 'decision_tree_model.joblib')
        print("Decision Tree model saved as 'decision_tree_model.joblib'")

        if fitted_scaler:
            joblib.dump(fitted_scaler, 'robust_scaler.joblib')
            print("Robust Scaler saved as 'robust_scaler.joblib'")

        if fitted_imputer:
            joblib.dump(fitted_imputer, 'iterative_imputer.joblib')
            print("Iterative Imputer saved as 'iterative_imputer.joblib'")

        joblib.dump(feature_names, 'feature_names.joblib')
        print("Feature names saved as 'feature_names.joblib'")

    except Exception as e:
        print(f"Error saving objects: {e}. Please check permissions and path.")

    print("\n--- Machine Learning Training Pipeline Complete ---")