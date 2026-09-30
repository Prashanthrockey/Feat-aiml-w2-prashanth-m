import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def get_data(filepath='data.csv'):
    """Loads CSV data or creates dummy data if file is missing."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset from '{filepath}'.")
    except FileNotFoundError:
        print(f"'{filepath}' not found. Generating sample imbalanced dataset...")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'Age': np.random.randint(22, 60, size=n),
            'Experience': np.random.randint(1, 35, size=n),
            'Salary': np.random.randint(30000, 150000, size=n),
            'City': np.random.choice(['Bangalore', 'Mumbai', 'Delhi'], size=n),
            'Target': np.random.choice([0, 1], size=n, p=[0.85, 0.15])  # Imbalanced
        })
        df.to_csv(filepath, index=False)
        print(f"Saved sample data to '{filepath}'.")
    return df

def preprocess_and_split(df):
    """Fills missing values, handles categorical data, and splits into train/test."""
    df = df.copy()

    # Fill missing values for numerical columns
    for col in ['Age', 'Experience', 'Salary']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # One-hot encode categorical features ('City')
    if 'City' in df.columns:
        df = pd.get_dummies(df, columns=['City'], drop_first=True)

    # Separate features and target label
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Train/Test Split (Stratified to maintain class proportions before SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def main():
    # 1. Load Data
    df = get_data('data.csv')

    # 2. Preprocess & Split
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    print("\n--- Before SMOTE ---")
    print(f"Training Class Counts:\n{y_train.value_counts().to_dict()}")

    # 3. Apply SMOTE only on Training Data (Prevents Data Leakage)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n--- After SMOTE ---")
    print(f"Resampled Training Class Counts:\n{y_train_res.value_counts().to_dict()}")

    # 4. Model Training
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_res, y_train_res)

    # 5. Evaluation on Original Test Set
    y_pred = clf.predict(X_test)

    print("\n--- Evaluation Metrics ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()