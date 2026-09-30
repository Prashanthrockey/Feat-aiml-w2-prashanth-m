import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_regression


data = {
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha", "Arjun",
        "Kavya", "Rohan", "Ananya", "Vikram", "Neha"
    ],
    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female"
    ],
    "City": [
        "Bangalore", "Mumbai", "Delhi", "Bangalore", "Chennai",
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Mumbai"
    ],
    "Education": [
        "UG", "PG", "UG", "PhD", "PG",
        "UG", "PG", "PhD", "UG", "PG"
    ],
    "Study_Hours": [4, 7, 3, 8, 5, 9, 6, 8, 4, 7],
    "Attendance": [78, 92, 70, 95, 82, 96, 88, 94, 75, 90],
    "Assignments": [65, 88, 60, 92, 74, 95, 85, 90, 68, 87],
    "Previous_Score": [68, 85, 62, 91, 73, 94, 82, 89, 65, 84],
    "Sleep_Hours": [6, 7, 5, 8, 6, 8, 7, 7, 5, 7],
    "Internet_Hours": [3, 2, 5, 1, 4, 2, 3, 2, 5, 2],
    "Final_Score": [70, 88, 65, 94, 76, 96, 84, 92, 68, 87]
}

df = pd.DataFrame(data)

print("Original Data")
print(df)

label_encoder = LabelEncoder()
df["Gender_Encoded"] = label_encoder.fit_transform(df["Gender"])

print("\nLabel Encoding")
print(df[["Gender", "Gender_Encoded"]])

one_hot_encoder = OneHotEncoder(sparse_output=False)

city_encoded = one_hot_encoder.fit_transform(df[["City"]])

city_columns = one_hot_encoder.get_feature_names_out(["City"])

city_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print("\nOne Hot Encoding")
print(city_df)

education_order = [["UG", "PG", "PhD"]]

ordinal_encoder = OrdinalEncoder(categories=education_order)

df["Education_Encoded"] = ordinal_encoder.fit_transform(
    df[["Education"]]
)

print("\nOrdinal Encoding")
print(df[["Education", "Education_Encoded"]])

print("\nEncoding Trade-offs")
print("LabelEncoder is simple but should mainly be used for target labels.")
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

print("\nOriginal Numerical Data")
print(X)

print("\nStandardScaler")
print(standard_df)

print("\nMinMaxScaler")
print(minmax_df)

print("\nRobustScaler")
print(robust_df)

print("\nScaling Trade-offs")
print("StandardScaler centers data around zero and works well for normally distributed data.")
print("MinMaxScaler changes values to a range between 0 and 1.")
print("RobustScaler is less affected by outliers because it uses the median and IQR.")


plt.figure(figsize=(8, 5))
plt.hist(X["Study_Hours"], bins=6)
plt.title("Study Hours Before Scaling")
plt.xlabel("Study Hours")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(standard_df["Study_Hours"], bins=6)
plt.title("Study Hours After StandardScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(minmax_df["Study_Hours"], bins=6)
plt.title("Study Hours After MinMaxScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(robust_df["Study_Hours"], bins=6)
plt.title("Study Hours After RobustScaler")
plt.xlabel("Scaled Study Hours")
plt.ylabel("Frequency")
plt.show()


X_features = df[numeric_features]
y = df["Final_Score"]

selector = SelectKBest(score_func=f_regression, k=5)

X_selected = selector.fit_transform(X_features, y)

selected_features = X_features.columns[selector.get_support()]

scores = selector.scores_

feature_scores = pd.DataFrame({
    "Feature": X_features.columns,
    "Score": scores
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Selection using SelectKBest")

print("\nFeature Scores")
print(feature_scores)

print("\nTop 5 Features")
for feature in selected_features:
    print(feature)

print("\nWhy the Top Features Matter")

reasons = {
    "Study_Hours": "Study time can directly affect preparation and performance.",
    "Attendance": "Regular attendance can provide better exposure to lessons and activities.",
    "Assignments": "Assignment performance reflects regular practice and understanding.",
    "Previous_Score": "Previous academic performance can indicate the student's learning level.",
    "Sleep_Hours": "Adequate sleep can support concentration and learning.",
    "Internet_Hours": "Internet usage can have different effects depending on how it is used."
}

for feature in selected_features:
    print(feature + ": " + reasons[feature])

print("\nFinal Selected Dataset")
selected_df = pd.DataFrame(
    X_selected,
    columns=selected_features
)

print(selected_df)
