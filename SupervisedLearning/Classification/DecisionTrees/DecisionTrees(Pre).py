import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, PowerTransformer # MODIFIED: Added PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import joblib # Used for saving/loading trained models and preprocessors
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer

# --- Global Variables for Fitted Objects ---
'''
These variables will store the trained model and fitted preprocessors.
Storing these globally allows them to be accessed and saved at the end of the pipeline.
This is crucial for ensuring that new, unseen data can be processed
identically during prediction, maintaining consistency between training and inference.
'''
fitted_scaler = None
trained_model = None
fitted_imputer = None
fitted_power_transformer = None # MODIFIED: Added global variable for PowerTransformer
lower_bound = None
upper_bound = None


# --- 1. Data Loading and Initial Inspection ---
'''
This is the foundational first step in any machine learning project.
The goal is to get a preliminary understanding of the raw dataset.
- Loading the CSV into a Pandas DataFrame.
- Creating a `.copy()` to ensure that the original DataFrame remains untouched
  throughout the preprocessing steps, preventing unintended modifications.
- `df.describe()` provides a statistical summary (count, mean, std, min, quartiles, max)
  of numerical columns, helping to spot data ranges, potential outliers, or inconsistencies.
- `df.info()` gives a concise summary including data types, non-null counts, and memory usage.
  This helps in identifying columns with missing values or incorrect data types.
- Converting all columns to `float64` ensures numerical precision and consistency
  for all subsequent mathematical operations and machine learning algorithms.
- `df.corr()` generates the correlation matrix, showing linear relationships between features.
  This is useful for early insights into feature dependencies.
'''
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
    
    Why this way (Rationale):
    - Exploratory Data Analysis (EDA) is crucial for building intuition about the data.
      Visualizations help confirm or challenge assumptions from `describe()` and `info()`.
    - Histograms: Show the distribution of individual features. This helps identify
      skewness (e.g., if data is concentrated on one side), multimodality, or sparsity.
    - Scatterplots (vs. index): Can reveal trends over the dataset's order or
      unexpected patterns that might indicate issues in data collection or structure.
    - Correlation Heatmap: Provides a quick visual summary of linear relationships
      between all pairs of features. Strong correlations might suggest multicollinearity
      or redundant features.
    - Boxplots/Violin Plots: Excellent for visualizing the spread, median, quartiles,
      and potential outliers for multiple numerical features side-by-side.
      Violin plots add density estimation to boxplots, showing value distribution.
    - These visualizations are performed *before* any transformations (imputation, scaling,
      outlier removal) to showcase the raw state of the data, which is vital for documentation
      and understanding initial data characteristics.
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
    
    Workflow Rationale:
    - Missing values MUST be handled before most machine learning models can be trained,
      as they typically cannot work with NaN values.
    - IterativeImputer (also known as MICE - Multiple Imputation by Chained Equations)
      is chosen over simpler methods (like mean/median imputation) because it's
      more sophisticated. It models each feature with missing values as a function
      of other features in a round-robin fashion, providing more accurate estimates.
    - This step occurs relatively early in the pipeline because accurate imputation
      benefits from having as much context from other features as possible. Also,
      outlier detection and scaling methods can be sensitive to missing values.
    - The imputer is `fitted` on the data (even if no missing values are found in this run)
      and stored globally, ensuring that if new data with missing values arrives for prediction,
      the exact same imputation logic is applied.
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
    This function is for reporting and does not remove outliers at this stage.
    
    Workflow Rationale:
    - This step serves as a diagnostic. It's important to understand the prevalence
      of extreme values before deciding on a removal or capping strategy.
    - It's done AFTER imputation, as missing values could skew outlier calculations
      or appear as pseudo-outliers.
    - This informational analysis helps to confirm whether the data contains
      "true" outliers or if extreme values are expected given the feature's distribution.
    - Various methods (zero values, IQR, Z-score) are used to provide a comprehensive view.
      Zero values can be significant for certain features (e.g., `chlorides` might genuinely be zero).
      IQR-based (Tukey's fences) and Z-score (standard deviations from mean) are common statistical measures.
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
    and performs power transformations on specified features.
    The fitted scaler and power transformer are stored globally.
    
    Workflow Rationale:
    - Initial Boxplot: Visualizes feature distributions *before* outlier filtering/scaling.
      This serves as a baseline to observe the impact of the subsequent steps.
    - Outlier Filtering (Isolation Forest): Isolation Forest is an unsupervised anomaly detection
      algorithm. It's particularly effective for high-dimensional data and doesn't make
      assumptions about data distribution. Filtering outliers *before* scaling is a best practice
      because extreme values can heavily influence the calculation of mean, standard deviation,
      or even IQR, distorting the scaling process. By removing these anomalies, the subsequent
      scaling becomes more accurate and representative of the 'normal' data.
    - Robust Scaling (`RobustScaler`): Applied after outlier filtering. `RobustScaler` is chosen
      because it scales features using the median and Interquartile Range (IQR) rather than
      mean and standard deviation (like `StandardScaler`). This makes it inherently **robust to outliers**
      that might still be present or that `IsolationForest` decided to keep. Scaling is critical
      for many ML algorithms (especially distance-based ones or those using gradient descent)
      to ensure all features contribute equally, preventing features with larger numerical ranges
      from dominating the model.
    - Power Transformation (`PowerTransformer` with `yeo-johnson`):
      MODIFIED: Replaced `np.log1p` with `PowerTransformer`.
      - **Why the change:** Your previous `np.log1p` encountered `inf` or `NaN` errors because
        it's undefined for values <= 0. After `RobustScaler`, it's possible that some scaled
        values became negative or zero. `PowerTransformer` (specifically the `yeo-johnson` method)
        is designed to handle a wider range of input values, including zeros and negatives. It
        automatically finds the optimal power parameter to transform data towards a more
        Gaussian (normal) distribution. Many ML models assume or perform better with normally
        distributed data. This step improves the model's ability to learn from skewed features.
      - `standardize=False` is used to preserve the scaling already applied by `RobustScaler`.
    - Debugging Checks: Added print statements to confirm that no NaNs or Infs remain after
      all transformations. This is a critical quality control step.
    - Last-resort `dropna`: A final safety measure to remove any remaining non-finite values.
      Ideally, `PowerTransformer` should prevent this, but it acts as a safeguard against
      unforeseen issues in the data.
    - Final Boxplot: Visualizes the data *after* all preprocessing, allowing for a comparison
      to the initial state and confirming the effects of scaling and transformation. 
    """
    print("\n--- Handling Outliers and Scaling Data ---")
    global fitted_scaler, fitted_power_transformer # MODIFIED: Added fitted_power_transformer to global scope

    # Initial boxplot before filtering
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_input.drop(columns='quality', errors='ignore')) # Exclude target for visualization
    plt.title("Boxplot Before Isolation Forest Filtering and Scaling")
    plt.tight_layout()
    #plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_pre_isoforest.png', dpi=300)
    plt.close()

    # Outlier detection and filtering using Isolation Forest
    isolated_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
    df_input["outlier_score"] = isolated_forest.fit_predict(df_input)
    df_filtered = df_input[df_input["outlier_score"] == 1].copy() # Keep inliers only

    # Identify features for scaling (exclude target and outlier_score)
    features_to_process = df_filtered.drop(columns=["quality", "outlier_score"], errors='ignore').columns

    # Apply Robust Scaling - fit on filtered data, transform it
    scaler = RobustScaler()
    fitted_scaler = scaler.fit(df_filtered[features_to_process]) # Fit and store the scaler
    df_scaled_array = fitted_scaler.transform(df_filtered[features_to_process])

    # Convert scaled features back to DataFrame
    df_processed = pd.DataFrame(df_scaled_array, columns=features_to_process, index=df_filtered.index)

    # MODIFIED START: Replaced log1p with PowerTransformer
    # Apply Power Transformation for potentially skewed features (post-scaling)
    # PowerTransformer handles values that are non-positive or zero robustly.
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    fitted_power_transformer = power_transformer # Store globally

    cols_for_power_transform = ["total sulfur dioxide", "residual sugar", "chlorides"]
    for col in cols_for_power_transform:
        if col in df_processed.columns:
            # Reshape the single column to a 2D array as required by fit_transform
            df_processed[col] = power_transformer.fit_transform(df_processed[[col]])[:,0]
    # MODIFIED END

    # Re-add 'quality' and 'outlier_score' to the processed DataFrame
    df_processed["quality"] = df_filtered["quality"]
    df_processed["outlier_score"] = df_filtered["outlier_score"]

    # --- Debugging checks after all transformations ---
    print("\n--- Checking for NaNs/Infs after all transformations ---")
    print("NaNs in df_processed (final):\n", df_processed.isnull().sum())
    print("Infs in df_processed (final):\n", df_processed.isin([np.inf, -np.inf]).sum()) 

    # If any NaNs/Infs still exist, convert to NaN and drop rows as a last resort
    initial_rows = len(df_processed)  
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.dropna(inplace=True)
    if len(df_processed) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df_processed)} rows containing NaNs or Infs after transformations.")


    # Final boxplot of processed data
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=45, ha="right", fontsize=10)
    sns.boxplot(data=df_processed.drop(columns=["quality", "outlier_score"], errors='ignore'))
    plt.title("Boxplot After Outlier Handling, Scaling, and Transformations")
    plt.tight_layout()
    #plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/boxplot_final.png', dpi=300)
    plt.close()

    print("Outlier handling and scaling complete.")
    return df_processed # Return the fully processed DataFrame


# --- 6. Model Training and Evaluation ---
def train_and_evaluate_model(df_processed):
    """
    Trains a Decision Tree Classifier, evaluates its performance,
    and plots the decision trees. The trained model is stored globally.
    
    Workflow Rationale:
    - Define Features (X) and Target (y): Clearly separating predictors from the variable
      to be predicted (`quality`). `outlier_score` is also excluded as it's a transient column.
    - Train-Test Split: This is a cornerstone of robust machine learning. It splits the data
      into a training set (used to train the model) and a test set (unseen data used for evaluation).
      This prevents overfitting, where a model performs well on training data but poorly on new data.
      `random_state` ensures reproducibility of the split.
    - Initialize and Train Model: `DecisionTreeClassifier` is a powerful and interpretable
      classification algorithm. `random_state` ensures reproducibility of the tree's structure.
      The model is `fitted` (trained) on the `X_train` and `y_train` data.
    - Plot Decision Trees: Visualizing the tree helps in understanding the decision rules
      the model has learned. A full tree can be complex, so a sample tree provides
      a more readable illustration.
    - Model Evaluation:
      - `accuracy_score`: Provides a simple measure of overall correct predictions.
      - `classification_report`: More detailed, showing precision, recall, and F1-score
        for each class. This is crucial for understanding performance on imbalanced classes.
      - `confusion_matrix`: Shows the counts of true positives, true negatives, false positives,
        and false negatives, allowing for a deeper analysis of specific prediction errors.
    - Return Feature Names: Essential for consistent prediction in deployment, ensuring
      that input data for prediction has features in the same order as the training data.
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
    #plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/full_dt.png', dpi=300)
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
    ##plt.savefig(f'/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots/sample_dt.png', dpi=300)
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
    '''
    Saving the fitted model and all preprocessors is crucial for deployment.
    This ensures that when you get new, unseen data for prediction, you can:
    1. Load these saved objects.
    2. Apply the exact same preprocessing steps (imputation, scaling, transformation)
       that were learned from the training data.
    3. Feed the consistently processed new data to the trained model for accurate predictions.
    `joblib` is preferred over `pickle` for large NumPy arrays and scikit-learn objects
    due to its efficiency.
    '''
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

        if fitted_power_transformer: # MODIFIED: Saving the PowerTransformer
            joblib.dump(fitted_power_transformer, 'power_transformer.joblib')
            print("Power Transformer saved as 'power_transformer.joblib'")

        joblib.dump(feature_names, 'feature_names.joblib')
        print("Feature names saved as 'feature_names.joblib'")

    except Exception as e:
        print(f"Error saving objects: {e}. Please check permissions and path.")

    print("\n--- Machine Learning Training Pipeline Complete ---")