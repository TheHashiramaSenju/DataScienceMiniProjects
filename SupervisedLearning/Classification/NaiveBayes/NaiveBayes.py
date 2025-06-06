import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler # Good for MultinomialNB/BernoulliNB if used
from scipy.stats import boxcox # For transforming skewed data for GaussianNB

# --- Configuration ---
DATA_PATH = '/home/notshadow/Documents/MiscFiles/Datascience/SupervisedLearning/Regression/LogisticRegression/diabetes.csv'
OUTCOME_COL = 'Outcome'
# Columns where 0 represents missing data (and should be replaced)
ZERO_AS_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# Columns that are naturally categorical/discrete (and can have 0)
NATURAL_ZERO_COLS = ['Pregnancies'] # 'Outcome' is also naturally 0 or 1

# --- Data Loading and Initial Exploration ---
def load_data(path):
    df = pd.read_csv(path)
    print("--- Original Data Head ---")
    print(df.head())
    print("\n--- Original Data Info ---")
    df.info()
    print("\n--- Original Data Description ---")
    print(df.describe())
    return df

# Data Cleaning ---
def clean_data(df_raw):
    df_clean = df_raw.copy()

    print("\n--- Data Cleaning Process ---")

    # 1. Handle zeros that represent missing values
    for col in ZERO_AS_NAN_COLS:
        # Replace 0 with NaN first for proper mean calculation
        df_clean[col] = df_clean[col].replace(0, np.nan)
        # Impute NaN values with the mean of the column
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        print(f"Replaced zeros/NaNs in '{col}' with mean: {df_clean[col].mean():.2f}")

    # 2. Convert all feature columns to float64 for consistency (Outcome can remain int)
    # Exclude the Outcome column from type conversion for features
    feature_cols = [col for col in df_clean.columns if col != OUTCOME_COL]
    for col in feature_cols:
        df_clean[col] = df_clean[col].astype('float64')

    print("\n--- Cleaned Data Description (after zero/NaN handling) ---")
    print(df_clean.describe())
    return df_clean

# --- Data Visualization ---
def visualize_data(df_cleaned):
    print("\n--- Data Visualization ---")

    # Box plots for outlier visualization
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_cleaned.drop(columns=[OUTCOME_COL])) # Drop outcome for feature box plots
    plt.title('Box Plot of Features (After Cleaning Zeros)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('box_plot_cleaned_data.png')
    plt.show()
    plt.close()

    # Distribution plots for all features
    df_cleaned.drop(columns=[OUTCOME_COL]).hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle('Histograms of Features (After Cleaning Zeros)', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('histograms_cleaned_data.png')
    plt.show()
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Cleaned Data')
    plt.tight_layout()
    plt.savefig('correlation_heatmap_cleaned_data.png')
    plt.show()
    plt.close()


# --- Naive Bayes Model Training and Evaluation ---
def train_and_evaluate_naive_bayes(X, y):
    print("\n--- Naive Bayes Model Training and Evaluation ---")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Using stratify=y ensures that the proportion of output classes is the same in train and test sets.

    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        # "Multinomial Naive Bayes": MultinomialNB(), # Typically for count data, needs non-negative features
        # "Bernoulli Naive Bayes": BernoulliNB()     # Typically for binary features
    }

    # For MultinomialNB and BernoulliNB, data often needs to be scaled to [0,1] or binarized
    # For this dataset, GaussianNB is the most suitable as features are continuous and normally distributed.
    # If you wanted to try MultinomialNB, you might need MinMaxScaler:
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # models["Multinomial Naive Bayes"] = MultinomialNB()
    # models["Multinomial Naive Bayes"].fit(X_train_scaled, y_train)
    # ... evaluation for MultinomialNB

    # For demonstration, we will primarily focus on GaussianNB
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\nResults for {name}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        # Visualize Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
                    yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {name}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.show()
        plt.close()

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# --- Main Execution Flow ---
if __name__ == "__main__":
    df_raw = load_data(DATA_PATH)
    df_cleaned = clean_data(df_raw)

    # Separate features (X) and target (y)
    X = df_cleaned.drop(columns=[OUTCOME_COL])
    y = df_cleaned[OUTCOME_COL]

    visualize_data(df_cleaned)
    train_and_evaluate_naive_bayes(X, y)

    print("\n--- Script Finished ---")