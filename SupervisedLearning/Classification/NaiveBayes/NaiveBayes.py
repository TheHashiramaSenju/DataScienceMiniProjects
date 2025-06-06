import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving/loading models and preprocessing assets

# --- Configuration ---
DATA_PATH = '/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Regression/LogisticRegression/diabetes.csv'
OUTCOME_COL = 'Outcome'
# Columns where 0 represents missing data (and should be replaced by mean)
ZERO_AS_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# --- 1. Data Loading and Initial Exploration ---
def load_and_explore_data(path):
    df = pd.read_csv(path)
    print("--- 1. Data Loading and Initial Exploration ---")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Information:")
    df.info()
    print("\nDataset Descriptive Statistics:")
    print(df.describe())
    return df

# --- 2. Data Cleaning and Preprocessing ---
def clean_and_preprocess_data(df_raw):
    df_clean = df_raw.copy()
    print("\n--- 2. Data Cleaning and Preprocessing ---")

    imputation_means = {}
    for col in ZERO_AS_NAN_COLS:
        # Replace 0 with NaN for proper mean calculation (as 0 is not a valid measure for these)
        df_clean[col] = df_clean[col].replace(0, np.nan)
        # Store the mean before imputation for consistent future preprocessing
        imputation_means[col] = df_clean[col].mean()
        # Impute NaN values with the calculated mean
        df_clean[col] = df_clean[col].fillna(imputation_means[col])
        print(f"Replaced zeros/NaNs in '{col}' with mean: {imputation_means[col]:.2f}")

    # Ensure all feature columns are float64 for consistency
    feature_cols = [col for col in df_clean.columns if col != OUTCOME_COL]
    for col in feature_cols:
        df_clean[col] = df_clean[col].astype('float64')

    print("\nCleaned Data Descriptive Statistics:")
    print(df_clean.describe())

    # Save imputation means for consistent preprocessing of new data
    joblib.dump(imputation_means, 'imputation_means.pkl')
    print("Imputation means saved to 'imputation_means.pkl' for future use.")

    return df_clean

# --- 3. Data Visualization ---
def visualize_data(df_cleaned):
    print("\n--- 3. Data Visualization ---")

    # Box plots for outlier visualization (after zero imputation)
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_cleaned.drop(columns=[OUTCOME_COL]))
    plt.title('Box Plot of Features (Post Zero Imputation)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visuals/box_plot_cleaned_data.png')
    plt.show()
    plt.close()

    # Distribution plots for all features
    df_cleaned.drop(columns=[OUTCOME_COL]).hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle('Histograms of Features (Post Zero Imputation)', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('visuals/histograms_cleaned_data.png')
    plt.show()
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Cleaned Data')
    plt.tight_layout()
    plt.savefig('visuals/correlation_heatmap_cleaned_data.png')
    plt.show()
    plt.close()

# --- 4. Model Training and Rigorous Evaluation ---
def train_and_evaluate_model(X, y):
    print("\n--- 4. Model Training and Rigorous Evaluation ---")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Outcome distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Outcome distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # Initialize and train Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("\nGaussian Naive Bayes model trained successfully.")

    # Evaluate model on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) # Get probabilities

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
                yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Gaussian Naive Bayes')
    plt.tight_layout()
    plt.savefig('visuals/confusion_matrix_gaussian_nb.png')
    plt.show()
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # Cross-validation for robust performance estimate
    print("\n--- Cross-Validation Performance ---")
    cv_accuracies = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_f1_weighted = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    print(f"5-Fold Cross-Validation Accuracies: {cv_accuracies}")
    print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies)*2:.4f})")
    print(f"Mean CV F1-Score (weighted): {np.mean(cv_f1_weighted):.4f}")

    # Save the trained model for deployment
    joblib.dump(model, 'diabetes_model.pkl')
    print("\nTrained model saved to 'diabetes_model.pkl'.")

    return model

# --- 5. Practical Application: Predicting New Patient Data ---
def predict_new_patient(model, new_patient_data_dict):
    print("\n--- 5. Practical Application: Predicting New Patient Data ---")

    # Load previously saved imputation means for consistent preprocessing
    try:
        imputation_means = joblib.load('imputation_means.pkl')
    except FileNotFoundError:
        print("Error: 'imputation_means.pkl' not found. Please run training script first.")
        return

    # Create a DataFrame from the new patient data dictionary
    new_patient_df = pd.DataFrame([new_patient_data_dict])
    print("\nNew Patient Data Received:")
    print(new_patient_df)

    # Preprocess the new data using the same steps as training
    processed_patient_df = new_patient_df.copy()
    for col in ZERO_AS_NAN_COLS:
        if col in processed_patient_df.columns:
            processed_patient_df[col] = processed_patient_df[col].replace(0, np.nan)
            processed_patient_df[col] = processed_patient_df[col].fillna(imputation_means[col])
    
    # Ensure all feature columns are float64
    for col in processed_patient_df.columns:
        processed_patient_df[col] = processed_patient_df[col].astype('float64')

    print("\nNew Patient Data After Preprocessing:")
    print(processed_patient_df)

    # Make prediction and get probabilities
    prediction = model.predict(processed_patient_df)[0]
    prediction_proba = model.predict_proba(processed_patient_df)[0]

    print("\n--- Prediction Results for New Patient ---")
    print(f"Predicted Diabetes Outcome (0=No, 1=Yes): {int(prediction)}")
    print(f"Probability of No Diabetes (0): {prediction_proba[0]:.4f}")
    print(f"Probability of Diabetes (1): {prediction_proba[1]:.4f}")

    if prediction == 1:
        print(f"\n**Recommendation:** Based on the data, the model suggests a HIGH likelihood of diabetes (Probability: {prediction_proba[1]:.2%}).")
        print("Further medical evaluation and diagnostic tests are strongly recommended.")
    else:
        print(f"\n**Recommendation:** Based on the data, the model suggests a LOW likelihood of diabetes (Probability: {prediction_proba[1]:.2%}).")
        print("Continue regular check-ups and maintain a healthy lifestyle.")

# --- Main Execution ---
if __name__ == "__main__":
    # Create a directory for visuals if it doesn't exist
    import os
    if not os.path.exists('visuals'):
        os.makedirs('visuals')

    # Step 1-4: Data Loading, Cleaning, Visualization, Training & Evaluation
    df_raw = load_and_explore_data(DATA_PATH)
    df_cleaned = clean_and_preprocess_data(df_raw)

    X = df_cleaned.drop(columns=[OUTCOME_COL])
    y = df_cleaned[OUTCOME_COL]

    visualize_data(df_cleaned)
    trained_model = train_and_evaluate_model(X, y)

    # Step 5: Demonstrate Practical Application with a New Patient
    print("\n" + "="*80)
    print("DEMONSTRATING PRACTICAL APPLICATION WITH A NEW PATIENT EXAMPLE")
    print("="*80 + "\n")

    # Example new patient data (replace with real data or user input in a deployed system)
    # Ensure these features match the exact names and order of your training features
    new_patient_example = {
        'Pregnancies': 2,
        'Glucose': 130,
        'BloodPressure': 70,
        'SkinThickness': 35,
        'Insulin': 0, # Example of a zero that needs imputation
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 45
    }

    # You would typically load the model here if this were a separate script
    # For this combined script, we use the `trained_model` object directly
    predict_new_patient(trained_model, new_patient_example)

    print("\n--- End of Diabetes Prediction Model Script ---")