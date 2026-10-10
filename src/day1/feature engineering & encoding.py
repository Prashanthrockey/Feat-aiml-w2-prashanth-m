import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Sample dataset
df = pd.DataFrame({
    'City': ['Bengaluru', 'Mumbai', 'Delhi', 'Bengaluru', 'Mumbai'],
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'Tier': ['Bronze', 'Gold', 'Silver', 'Gold', 'Bronze'],
    'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
})

print("Original dataset:")
print(df)

# 1. LabelEncoder on target variable only
le = LabelEncoder()
df['Churn_Encoded'] = le.fit_transform(df['Churn'])

# 2. OneHotEncoder on nominal column (City)
ohe = OneHotEncoder(drop='first', sparse_output=False)
city_encoded = ohe.fit_transform(df[['City']])
city_df = pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['City']))

# 3. OrdinalEncoder on ranked columns
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
tier_order = ['Bronze', 'Silver', 'Gold']

oe = OrdinalEncoder(categories=[education_order, tier_order])
df[['Education_Encoded', 'Tier_Encoded']] = oe.fit_transform(df[['Education', 'Tier']])

# Final combined dataframe
final_df = pd.concat([df[['Education_Encoded', 'Tier_Encoded', 'Churn_Encoded']], city_df], axis=1)

print("\nFinal encoded dataset:")
print(final_df)