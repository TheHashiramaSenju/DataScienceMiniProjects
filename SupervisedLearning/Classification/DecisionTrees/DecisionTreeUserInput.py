import pandas as pd
import numpy as np
import joblib
import sys # For exiting if files not found

# --- Paths to your saved artifacts ---
MODEL_PATH = 'decision_tree_model.joblib'
SCALER_PATH = 'robust_scaler.joblib'
IMPUTER_PATH = 'iterative_imputer.joblib' # If you implemented it
FEATURE_NAMES_PATH = 'feature_names.joblib'

def load_artifacts():
    """Loads the trained model, scaler, imputer, and feature names."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded Decision Tree model from {MODEL_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded Robust Scaler from {SCALER_PATH}")
        # Only load imputer if it was actually saved
        try:
            imputer = joblib.load(IMPUTER_PATH)
            print(f"Loaded Iterative Imputer from {IMPUTER_PATH}")
        except FileNotFoundError:
            imputer = None # No imputer found, handle gracefully
            print("Iterative Imputer not found (might not have been saved).")
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        print(f"Loaded feature names from {FEATURE_NAMES_PATH}")
        return model, scaler, imputer, feature_names
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Make sure you ran the training script. {e}")
        sys.exit(1) # Exit the program if essential files are missing
    except Exception as e:
        print(f"An unexpected error occurred while loading artifacts: {e}")
        sys.exit(1)

def get_user_input(feature_names):
    """
    Prompts the user to input values for each feature.
    Includes a placeholder for 'outlier_score' for consistency.
    """
    print("\nPlease enter the following details for the wine sample:")
    user_data = {}
    for feature in feature_names:
        # Loop until valid numerical input is received
        while True:
            try:  
                value = float(input(f"Enter value for '{feature}': "))
                user_data[feature] = value
                break # Exit loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
    # Add outlier_score. When making predictions on new data, if you filtered out outliers,
    # you're implicitly assuming the new data is 'inlier'. So, set to 1.
    user_data['outlier_score'] = 1
    return user_data

def preprocess_new_data(raw_data, feature_names, scaler, imputer):
    """
    Applies the same preprocessing steps as the training data.
    IMPORTANT: The order and type of transformations MUST match the training pipeline.
    """
    print("\nPreprocessing new data...")

    # 1. Convert to DataFrame and ensure correct column order
    # Create DataFrame from raw_data, ensuring all feature_names are present.
    # Use `columns` argument explicitly to maintain order and add 'outlier_score'
    # The `raw_data` dictionary contains all features from `feature_names` plus 'outlier_score'.
    # We create a dataframe with the exact columns that the model expects, which
    # are derived from feature_names + 'outlier_score' if it was part of the training input.
    # The model was trained on X = df_processed.drop(columns=["quality", "outlier_score"]),
    # so we need to construct a DataFrame that matches X.columns.
    # 'outlier_score' is not a feature for the model, but it was part of the original df_processed,
    # and its removal is handled by .drop().
    # Here, we ensure the input DataFrame to scaler.transform is aligned with what scaler expects.

    # Build a DataFrame with only the features the scaler expects
    df_for_preprocessing = pd.DataFrame([raw_data])[feature_names]

    # 2. Impute missing values (if imputer was fitted and necessary)
    # This step is critical if your training data had missing values that were imputed.
    # The imputer should only be used if there are missing values in the *new* input,
    # or if the imputer was designed to also "transform" non-missing data in some way (rare for IterativeImputer).
    # For user input, we assume all values are provided, so imputation on *this specific input* is skipped.
    # However, if your training data had NA's and your imputer transformed data that wasn't NA, this would be crucial.
    # For IterativeImputer specifically, it only transforms NA's.
    # If the user *could* input NA, this step would be used:
    # if imputer:
    #    df_for_preprocessing_array = imputer.transform(df_for_preprocessing)
    #    df_for_preprocessing = pd.DataFrame(df_for_preprocessing_array, columns=feature_names)


    # 3. Apply Robust Scaling
    # Use the *loaded* scaler to transform the new data. DO NOT fit again.
    scaled_data_array = scaler.transform(df_for_preprocessing)
    df_scaled = pd.DataFrame(scaled_data_array, columns=feature_names)

    # 4. Apply Log Transformations (The "Why" explained below)
    # Why again log transformations?
    # Because your *training data* had these transformations applied.
    # The model learned patterns from data where "residual sugar", "chlorides",
    # and "total sulfur dioxide" had undergone np.log1p().
    # If you feed new data to the model without applying the *same* transformations,
    # the feature values will be on a different scale/distribution than what the model expects,
    # leading to incorrect predictions.
    # Example: If training data's 'residual sugar' values ranged from 0.1 to 1.5 after log1p,
    # but your new data's 'residual sugar' is 5.0 (without log1p), the model will interpret 5.0
    # as a very different value than it would if it were log1p'd to ~1.7.

    if "total sulfur dioxide" in df_scaled.columns:
        df_scaled["total sulfur dioxide"] = np.log1p(df_scaled["total sulfur dioxide"])
    if "residual sugar" in df_scaled.columns:
        df_scaled["residual sugar"] = np.log1p(df_scaled["residual sugar"])
    if "chlorides" in df_scaled.columns:
        df_scaled["chlorides"] = np.log1p(df_scaled["chlorides"])

    print("Data preprocessing complete.")
    return df_scaled

def make_prediction(model, preprocessed_data):
    """Makes a prediction using the loaded model."""
    print("Making prediction...")
    # The model.predict() method expects a 2D array or DataFrame, even for a single sample.
    predicted_quality = model.predict(preprocessed_data)
    return predicted_quality[0] # Get the first (and only) prediction

if __name__ == "__main__":
    print("--- Starting Wine Quality Prediction Service ---")
    # Load all necessary components
    model, scaler, imputer, feature_names = load_artifacts()

    # Get dynamic input from the user
    raw_user_input = get_user_input(feature_names)

    # Preprocess the user's input data
    preprocessed_user_data = preprocess_new_data(raw_user_input, feature_names, scaler, imputer)

    # Make a prediction
    final_prediction = make_prediction(model, preprocessed_user_data)

    print(f"\n--- Prediction Result ---")
    print(f"The predicted wine quality is: {int(final_prediction)}") # Convert to int for quality rating
    print("-------------------------")