import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_titanic_data(filepath='titanic.csv'):
    """Loads Titanic dataset or creates a sample DataFrame if file is missing."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset from '{filepath}'.")
    except FileNotFoundError:
        print(f"'{filepath}' not found. Creating synthetic Titanic sample dataset...")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'PassengerId': range(1, n + 1),
            'Pclass': np.random.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5]),
            'Sex': np.random.choice(['male', 'female'], size=n),
            'Age': np.random.choice([np.nan, 22, 38, 26, 35, 54, 2, 27], size=n),
            'SibSp': np.random.choice([0, 1, 2, 3], size=n, p=[0.7, 0.2, 0.07, 0.03]),
            'Parch': np.random.choice([0, 1, 2], size=n, p=[0.8, 0.15, 0.05]),
            'Fare': np.random.choice([7.25, 71.28, 7.92, 53.10, 8.05], size=n),
            'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], size=n, p=[0.7, 0.2, 0.08, 0.02]),
            'Survived': np.random.choice([0, 1], size=n, p=[0.62, 0.38])
        })
        df.to_csv(filepath, index=False)
    return df

def perform_eda(df):
    """Prints basic Exploratory Data Analysis statistics."""
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values per Column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

def engineer_features(df):
    """Engineers new features like FamilySize and IsAlone."""
    df = df.copy()
    
    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    return df

def build_preprocessing_pipeline(num_features, cat_features):
    """Creates a ColumnTransformer pipeline for imputation, scaling, and encoding."""
    
    # Numerical pipeline: Impute missing with median -> StandardScale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing with most frequent -> OneHotEncode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine feature transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    return preprocessor

def main():
    # 1. Load Data
    raw_df = load_titanic_data('titanic.csv')
    
    # 2. EDA
    perform_eda(raw_df)
    
    # 3. Feature Engineering
    df = engineer_features(raw_df)
    
    # Define target and feature columns
    target_col = 'Survived'
    drop_cols = ['PassengerId', target_col]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_col]
    
    # Identify numerical and categorical column names
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 4. Split into Train / Test before fitting pipelines
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Build and Fit Preprocessing Pipeline
    preprocessor = build_preprocessing_pipeline(num_features, cat_features)
    
    # Fit on training data and transform train & test sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Retrieve engineered column names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_cols
    
    # 6. Reconstruct ML-ready DataFrames
    train_df_processed = pd.DataFrame(X_train_processed, columns=all_feature_names)
    train_df_processed['Target'] = y_train.values
    
    # 7. Save Export Output
    output_filename = 'ml_ready_features.csv'
    train_df_processed.to_csv(output_filename, index=False)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Processed features shape: {X_train_processed.shape}")
    print(f"Exported ML-ready dataset to '{output_filename}'.")

if __name__ == '__main__':
    main()