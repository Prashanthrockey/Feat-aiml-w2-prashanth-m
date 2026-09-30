import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_regression

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(
    base_path,
    "data",
    "student performance.csv"
)

df = pd.read_csv(data_path)

print("Original Data:")
print(df)


label_encoder = LabelEncoder()
df["Gender_Encoded"] = label_encoder.fit_transform(df["Gender"])

print("\nLabel Encoding:")
print(df[["Gender", "Gender_Encoded"]])


one_hot_encoder = OneHotEncoder(sparse_output=False)

city_encoded = one_hot_encoder.fit_transform(df[["City"]])

city_columns = one_hot_encoder.get_feature_names_out(["City"])

city_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print("\nOne Hot Encoding:")
print(city_df)


education_order = [["UG", "PG", "PhD"]]

ordinal_encoder = OrdinalEncoder(categories=education_order)

df["Education_Encoded"] = ordinal_encoder.fit_transform(
    df[["Education"]]
)

print("\nOrdinal Encoding:")
print(df[["Education", "Education_Encoded"]])


print("\nEncoding Trade-offs:")
print("LabelEncoder is useful for converting categories into numbers.")
print("OneHotEncoder is useful for categories without a natural order.")
print("OrdinalEncoder is useful when categories have a meaningful order.")


numeric_features = [
    "Study_Hours",
    "Attendance",
    "Assignments",
    "Previous_Score",
    "Sleep_Hours",
    "Internet_Hours"
]

X = df[numeric_features]

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

standard_scaled = standard_scaler.fit_transform(X)
minmax_scaled = minmax_scaler.fit_transform(X)
robust_scaled = robust_scaler.fit_transform(X)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=numeric_features
)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=numeric_features
)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=numeric_features
)

print("\nStandardScaler:")
print(standard_df)

print("\nMinMaxScaler:")
print(minmax_df)

print("\nRobustScaler:")
print(robust_df)


print("\nScaling Trade-offs:")
print("StandardScaler centers the data around zero.")
print("MinMaxScaler scales values between 0 and 1.")
print("RobustScaler is less affected by outliers.")


plt.figure(figsize=(8, 5))
plt.hist(X["Study_Hours"], bins=6)
plt.title("Study Hours Before Scaling")
plt.xlabel("Study Hours")
plt.ylabel("Frequency")
plt.savefig("visualization/study_hours_before_scaling.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(standard_df["Study_Hours"], bins=6)
plt.title("Study Hours After StandardScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.savefig("visualization/study_hours_standard_scaler.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(minmax_df["Study_Hours"], bins=6)
plt.title("Study Hours After MinMaxScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.savefig("visualization/study_hours_minmax_scaler.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(robust_df["Study_Hours"], bins=6)
plt.title("Study Hours After RobustScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.savefig("visualization/study_hours_robust_scaler.png")
plt.show()


y = df["Final_Score"]

selector = SelectKBest(
    score_func=f_regression,
    k=5
)

X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Scores:")
print(feature_scores)

print("\nTop 5 Features:")

for feature in selected_features:
    print(feature)

print("\nSelected Features Dataset:")
selected_df = pd.DataFrame(
    X_selected,
    columns=selected_features
)

print(selected_df)