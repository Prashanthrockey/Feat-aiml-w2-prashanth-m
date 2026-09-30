import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression


np.random.seed(42)

data = pd.DataFrame({
    "Age": np.random.randint(21, 45, 100),
    "Experience": np.random.randint(0, 20, 100),
    "Salary": np.random.randint(25000, 150000, 100),
    "City": np.random.choice(
        ["Bangalore", "Chennai", "Hyderabad", "Mumbai", "Delhi"], 100
    ),
    "Education": np.random.choice(
        ["Bachelors", "Masters", "PhD"], 100
    )
})

print(data.head(10))
print(data.shape)
print(data.dtypes)


label = LabelEncoder()
data["City_Label"] = label.fit_transform(data["City"])

print("\nLabel Encoding")
print(data[["City", "City_Label"]].head())


onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

city_encoded = onehot.fit_transform(data[["City"]])

city_encoded = pd.DataFrame(
    city_encoded,
    columns=onehot.get_feature_names_out(["City"])
)

print("\nOne Hot Encoding")
print(city_encoded.head())


ordinal = OrdinalEncoder(
    categories=[["Bachelors", "Masters", "PhD"]]
)

data["Education_Level"] = ordinal.fit_transform(
    data[["Education"]]
)

print("\nOrdinal Encoding")
print(data[["Education", "Education_Level"]].head())


features = data[["Age", "Experience", "Salary"]]

standard = StandardScaler()
standard_data = standard.fit_transform(features)

standard_data = pd.DataFrame(
    standard_data,
    columns=features.columns
)

print("\nStandard Scaler")
print(standard_data.head())


minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(features)

minmax_data = pd.DataFrame(
    minmax_data,
    columns=features.columns
)

print("\nMin Max Scaler")
print(minmax_data.head())


robust = RobustScaler()
robust_data = robust.fit_transform(features)

robust_data = pd.DataFrame(
    robust_data,
    columns=features.columns
)

print("\nRobust Scaler")
print(robust_data.head())


plt.figure(figsize=(8, 5))
features.boxplot()
plt.title("Before Scaling")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
standard_data.boxplot()
plt.title("After Standard Scaling")
plt.tight_layout()
plt.show()


x = data[["Age", "Experience", "City_Label", "Education_Level"]]
y = data["Salary"]

selector = SelectKBest(score_func=f_regression, k="all")
selector.fit(x, y)

scores = pd.DataFrame({
    "Feature": x.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Selection")
print(scores)


top_features = scores.head(5)

print("\nTop 5 Features")

for _, row in top_features.iterrows():
    print(row["Feature"], ":", round(row["Score"], 2))


plt.figure(figsize=(8, 5))
plt.bar(scores["Feature"], scores["Score"])
plt.title("Feature Selection Scores")
plt.xlabel("Features")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


data.to_csv("feature_engineering_output.csv", index=False)
scores.to_csv("feature_scores.csv", index=False)

print("\nDay 2 completed successfully")