import numpy as np
import pandas as pd
# Fixed import line below: removed 'CrossVal'
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def get_data(filepath='data.csv'):
    """Loads CSV data or creates dummy data matching previous columns if missing."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset from '{filepath}'.")
    except FileNotFoundError:
        print(f"'{filepath}' not found. Generating sample dataset...")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'Age': np.random.randint(22, 60, size=n),
            'Experience': np.random.randint(1, 35, size=n),
            'Salary': np.random.randint(30000, 150000, size=n),
            'City': np.random.choice(['Bangalore', 'Mumbai', 'Delhi'], size=n),
            'Target': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        })
        df.to_csv(filepath, index=False)
    return df

def preprocess_data(df):
    """Clean missing values and encode categorical columns."""
    df = df.copy()

    # Handle missing numerical values
    for col in ['Age', 'Experience', 'Salary']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # One-hot encode categorical features
    if 'City' in df.columns:
        df = pd.get_dummies(df, columns=['City'], drop_first=True)

    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y

def evaluate_scalers(X_train, y_train):
    """Demonstrates cross-validation across different scalers using Pipelines."""
    scalers = {
        'StandardScaler (Mean=0, Std=1)': StandardScaler(),
        'MinMaxScaler (Bounded [0, 1])': MinMaxScaler(),
        'RobustScaler (Outlier Resistant)': RobustScaler()
    }

    print("\n--- 5-Fold Cross-Validation Scores ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, scaler in scalers.items():
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', LogisticRegression(random_state=42))
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        print(f"{name}: Mean F1-Score = {scores.mean():.4f} (+/- {scores.std():.4f})")

def main():
    # 1. Load and clean
    df = get_data('data.csv')
    X, y = preprocess_data(df)

    # 2. Train/Test Split (Holdout Set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    # 3. Perform Cross-Validation to select/compare scalers
    evaluate_scalers(X_train, y_train)

    # 4. Train final model with chosen scaler (StandardScaler) on full training set
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    final_pipeline.fit(X_train, y_train)

    # 5. Evaluate on Holdout Test Set
    y_pred = final_pipeline.predict(X_test)
    print("\n--- Final Test Set Evaluation ---")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()