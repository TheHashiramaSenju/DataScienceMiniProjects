import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import os # For managing directories

# --- Configuration ---
# File path for the diabetes dataset (adjust as needed)
DATA_PATH = '/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Regression/LogisticRegression/diabetes.csv'
OUTCOME_COL = 'Outcome' # The target variable column name
# Columns where '0' logically represents a missing value in the raw data
ZERO_AS_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# Directory to save visualizations and models
OUTPUT_DIR = 'diabetes_model_output'

# --- 1. Data Collection / Ingestion ---
def collect_data(file_path):
    """
    Loads the raw dataset from the specified file path.
    """
    print("\n--- 1. Data Collection / Ingestion ---")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        print("\nInitial Data Head:")
        print(df.head())
        print("\nInitial Data Info:")
        df.info()
        print("\nInitial Data Descriptive Statistics:")
        print(df.describe())
        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please check the path.")
        return None

# --- 2. Data Transformation / Preprocessing ---
def preprocess_data(df_raw):
    """
    Cleans and preprocesses the raw dataframe.
    Handles missing values (zeros in specific columns) by imputation with means.
    Saves the calculated imputation means for consistent future preprocessing.
    """
    print("\n--- 2. Data Transformation / Preprocessing ---")
    df_clean = df_raw.copy()

    imputation_means = {}
    for col in ZERO_AS_NAN_COLS:
        # Replace 0 with NaN for proper mean calculation
        df_clean[col] = df_clean[col].replace(0, np.nan)
        # Calculate mean from the current (training) data and store it
        imputation_means[col] = df_clean[col].mean()
        # Impute NaN values with the calculated mean
        df_clean[col] = df_clean[col].fillna(imputation_means[col])
        print(f"Processed column '{col}': Zeros/NaNs replaced with mean {imputation_means[col]:.2f}")

    # Ensure all feature columns are of float64 type for consistency
    feature_cols = [col for col in df_clean.columns if col != OUTCOME_COL]
    for col in feature_cols:
        df_clean[col] = df_clean[col].astype('float64')

    # Save imputation means for consistent preprocessing of new, unseen data
    joblib.dump(imputation_means, os.path.join(OUTPUT_DIR, 'imputation_means.pkl'))
    print(f"Imputation means saved to '{OUTPUT_DIR}/imputation_means.pkl'.")

    print("\nCleaned Data Descriptive Statistics (after preprocessing):")
    print(df_clean.describe())
    return df_clean

# --- 3. Exploratory Data Analysis (EDA) / Visualization ---
def perform_eda(df_processed):
    """
    Generates and saves key visualizations to understand the processed data.
    """
    print("\n--- 3. Exploratory Data Analysis (EDA) / Visualization ---")

    # Box plots for outlier visualization
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_processed.drop(columns=[OUTCOME_COL]))
    plt.title('Box Plot of Features (Post Preprocessing)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'box_plot_processed_data.png'))
    plt.close()
    print(f"Box plot saved to '{OUTPUT_DIR}/box_plot_processed_data.png'")

    # Histograms for feature distributions
    df_processed.drop(columns=[OUTCOME_COL]).hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle('Histograms of Features (Post Preprocessing)', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'histograms_processed_data.png'))
    plt.close()
    print(f"Histograms saved to '{OUTPUT_DIR}/histograms_processed_data.png'")

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Processed Data')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap_processed_data.png'))
    plt.close()
    print(f"Correlation heatmap saved to '{OUTPUT_DIR}/correlation_heatmap_processed_data.png'")

# --- 4. Model Development / Training ---
def train_model(X, y):
    """
    Trains the Gaussian Naive Bayes model.
    """
    print("\n--- 4. Model Development / Training ---")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: Training samples={len(X_train)}, Test samples={len(X_test)}")
    print(f"Outcome distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Outcome distribution in test set:\n{y_test.value_counts(normalize=True)}")

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("\nGaussian Naive Bayes model trained successfully.")

    # Save the trained model
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'diabetes_model.pkl'))
    print(f"Trained model saved to '{OUTPUT_DIR}/diabetes_model.pkl'.")

    return model, X_test, y_test # Return test set for evaluation

# --- 5. Model Evaluation ---
def evaluate_model(model, X_test, y_test, X_full, y_full):
    """
    Evaluates the trained model using various metrics and cross-validation.
    Generates and saves a confusion matrix.
    """
    print("\n--- 5. Model Evaluation ---")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Performance on Test Set
    print("\n--- Performance on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
                yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Gaussian Naive Bayes')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_gaussian_nb.png'))
    plt.close()
    print(f"Confusion matrix saved to '{OUTPUT_DIR}/confusion_matrix_gaussian_nb.png'")

    # Cross-Validation for Robustness
    print("\n--- Cross-Validation Performance (5-Fold) ---")
    cv_accuracies = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
    cv_f1_weighted = cross_val_score(model, X_full, y_full, cv=5, scoring='f1_weighted')
    cv_precision_weighted = cross_val_score(model, X_full, y_full, cv=5, scoring='precision_weighted')
    cv_recall_weighted = cross_val_score(model, X_full, y_full, cv=5, scoring='recall_weighted')

    print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies)*2:.4f})")
    print(f"Mean CV F1-Score (weighted): {np.mean(cv_f1_weighted):.4f}")
    print(f"Mean CV Precision (weighted): {np.mean(cv_precision_weighted):.4f}")
    print(f"Mean CV Recall (weighted): {np.mean(cv_recall_weighted):.4f}")

# --- 6. Model Deployment / Prediction Service ---
def load_prediction_assets():
    """
    Loads the trained model and preprocessing assets (imputation means).
    """
    print("\n--- Loading Prediction Assets ---")
    try:
        model = joblib.load(os.path.join(OUTPUT_DIR, 'diabetes_model.pkl'))
        imputation_means = joblib.load(os.path.join(OUTPUT_DIR, 'imputation_means.pkl'))
        print("Prediction model and preprocessing means loaded successfully.")
        return model, imputation_means
    except FileNotFoundError:
        print(f"Error: Model or imputation means not found in '{OUTPUT_DIR}'.")
        print("Please ensure you've run the training pipeline first.")
        return None, None

def preprocess_new_data_for_prediction(new_patient_df, imputation_means):
    """
    Applies the exact same preprocessing steps to new patient data as done during training.
    """
    processed_df = new_patient_df.copy()

    for col in ZERO_AS_NAN_COLS:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].replace(0, np.nan)
            processed_df[col] = processed_df[col].fillna(imputation_means[col])
    
    for col in processed_df.columns:
        processed_df[col] = processed_df[col].astype('float64')

    return processed_df

def predict_patient_risk(model, imputation_means, patient_data_dict):
    """
    Predicts diabetes risk for a single patient.
    """
    print("\n--- Predicting Risk for a Single Patient ---")
    new_patient_df = pd.DataFrame([patient_data_dict])
    processed_df = preprocess_new_data_for_prediction(new_patient_df, imputation_means)

    prediction = model.predict(processed_df)[0]
    prediction_proba = model.predict_proba(processed_df)[0]

    print(f"Patient Data Provided:\n{new_patient_df}")
    print(f"\nProcessed Data for Prediction:\n{processed_df}")
    print(f"\nPrediction Results:")
    print(f"Predicted Diabetes Outcome (0=No, 1=Yes): {int(prediction)}")
    print(f"Probability of No Diabetes (0): {prediction_proba[0]:.4f}")
    print(f"Probability of Diabetes (1): {prediction_proba[1]:.4f}")

    if prediction == 1:
        print(f"\n**Medical Suggestion:** This patient shows a HIGH likelihood of diabetes (Probability: {prediction_proba[1]:.2%}).")
        print("Strongly recommend immediate further diagnostic tests and consultation with an endocrinologist.")
    else:
        print(f"\n**Medical Suggestion:** This patient shows a LOW likelihood of diabetes (Probability: {prediction_proba[1]:.2%}).")
        print("Recommend regular follow-ups and emphasis on maintaining a healthy lifestyle.")

def predict_batch_risk(model, imputation_means, batch_data_df):
    """
    Predicts diabetes risk for a batch of patients.
    """
    print("\n--- Predicting Risk for a Batch of Patients ---")
    print(f"Received {len(batch_data_df)} patients for batch prediction.")

    processed_batch_df = preprocess_new_data_for_prediction(batch_data_df, imputation_means)
    
    predictions = model.predict(processed_batch_df)
    probabilities = model.predict_proba(processed_batch_df)

    # Add predictions and probabilities to the original batch DataFrame
    batch_data_df['Predicted_Outcome'] = predictions
    batch_data_df['Probability_No_Diabetes'] = probabilities[:, 0]
    batch_data_df['Probability_Diabetes'] = probabilities[:, 1]

    print("\nBatch Prediction Results (first 5 rows):")
    print(batch_data_df.head())
    print(f"\nTotal patients predicted: {len(batch_data_df)}")
    print(f"Predicted Diabetes cases: {batch_data_df['Predicted_Outcome'].sum()}")

    return batch_data_df

# --- 7. High-Level Methods (Application Scenarios) ---
def diabetes_risk_assessment_app():
    """
    A high-level function simulating a clinical application for single patient assessment.
    """
    model, imputation_means = load_prediction_assets()
    if model is None:
        return

    print("\n--- Diabetes Risk Assessment Application ---")
    print("Enter patient data (enter 'done' for any field to finish):")

    patient_data = {}
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    for feature in features:
        while True:
            user_input = input(f"Enter {feature}: ").strip().lower()
            if user_input == 'done':
                print("Exiting data entry.")
                return
            try:
                value = float(user_input)
                patient_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number or 'done'.")
    
    predict_patient_risk(model, imputation_means, patient_data)
    print("\n--- Risk Assessment Complete ---")

def community_screening_tool(csv_file_path):
    """
    A high-level function simulating a tool for community-wide diabetes screening.
    Reads a CSV, predicts risk for all individuals, and generates a report.
    """
    model, imputation_means = load_prediction_assets()
    if model is None:
        return

    print(f"\n--- Community Diabetes Screening Tool ({csv_file_path}) ---")
    try:
        community_df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(community_df)} records for screening.")
        
        # Ensure the CSV has all required features (except Outcome)
        required_features = [f for f in community_df.columns if f != OUTCOME_COL]
        missing_features = [f for f in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] if f not in required_features]
        if missing_features:
            print(f"Error: Missing required features in CSV: {missing_features}")
            return

        screened_results_df = predict_batch_risk(model, imputation_means, community_df)
        
        output_csv_path = os.path.join(OUTPUT_DIR, 'community_screening_results.csv')
        screened_results_df.to_csv(output_csv_path, index=False)
        print(f"\nCommunity screening results saved to '{output_csv_path}'.")

        # Basic Risk Stratification Report
        high_risk_count = screened_results_df[screened_results_df['Predicted_Outcome'] == 1].shape[0]
        total_count = len(screened_results_df)
        print(f"\n--- Community Screening Summary ---")
        print(f"Total individuals screened: {total_count}")
        print(f"Individuals predicted with diabetes risk: {high_risk_count} ({high_risk_count/total_count:.2%})")
        print("Consider prioritizing follow-ups for individuals with 'Predicted_Outcome' = 1.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}.")
    except Exception as e:
        print(f"An error occurred during community screening: {e}")

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("           DIABETES PREDICTION MODEL: END-TO-END PIPELINE         ")
    print("="*80)

    # Flow: Data Collection -> Transformation -> EDA -> Model Development -> Evaluation
    df_raw = collect_data(DATA_PATH)
    if df_raw is None:
        exit() # Exit if data loading failed

    df_processed = preprocess_data(df_raw)
    perform_eda(df_processed)

    X_full = df_processed.drop(columns=[OUTCOME_COL])
    y_full = df_processed[OUTCOME_COL]

    trained_model, X_test_eval, y_test_eval = train_model(X_full, y_full)
    evaluate_model(trained_model, X_test_eval, y_test_eval, X_full, y_full)

    print("\n" + "="*80)
    print("               DEMONSTRATING PRACTICAL APPLICATIONS              ")
    print("="*80 + "\n")

    # --- Demonstrate Scenario 1: Single Patient Risk Assessment ---
    print("\n--- Scenario 1: Interactive Single Patient Risk Assessment ---")
    # This will prompt you for input in the console.
    # To avoid this during an automated run, you can comment this out or use a predefined dict
    # diabetes_risk_assessment_app()

    # Predefined example for single patient demonstration (if you don't want interactive input)
    example_patient = {
        'Pregnancies': 2,
        'Glucose': 130,
        'BloodPressure': 70,
        'SkinThickness': 35,
        'Insulin': 0, # This zero will be imputed by the saved mean
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 45
    }
    loaded_model, loaded_imputation_means = load_prediction_assets()
    if loaded_model and loaded_imputation_means:
        predict_patient_risk(loaded_model, loaded_imputation_means, example_patient)


    # --- Demonstrate Scenario 2: Community Screening (Batch Prediction) ---
    print("\n--- Scenario 2: Community-Wide Diabetes Screening ---")
    # To demonstrate this, let's create a dummy CSV file that looks like new patient data
    dummy_community_data = {
        'Pregnancies': [1, 5, 0, 3],
        'Glucose': [110, 190, 95, 140],
        'BloodPressure': [72, 80, 60, 75],
        'SkinThickness': [20, 0, 15, 30], # Example with a zero
        'Insulin': [79, 0, 0, 150],       # Example with zeros
        'BMI': [28.1, 40.5, 22.0, 31.8],
        'DiabetesPedigreeFunction': [0.372, 0.9, 0.25, 0.55],
        'Age': [25, 50, 20, 35]
    }
    dummy_community_df = pd.DataFrame(dummy_community_data)
    dummy_csv_path = os.path.join(OUTPUT_DIR, 'dummy_community_screening_data.csv')
    dummy_community_df.to_csv(dummy_csv_path, index=False)
    print(f"Dummy community screening data created at '{dummy_csv_path}'.")

    community_screening_tool(dummy_csv_path)

    print("\n" + "="*80)
    print("           END OF DIABETES PREDICTION MODEL DEMONSTRATION         ")
    print("="*80)