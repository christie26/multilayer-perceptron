import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Column names
columns = ["id", "diagnosis"] + [f"feature{i}" for i in range(1, 31)]

# Load CSV (first 20 rows)
df = pd.read_csv("data.csv", names=columns)
df = df.head(20)

# Basic info
print(f"Data size: {df.shape}\n")
df.info()
df.describe()

# Check diagnosis value distribution
print("\nDiagnosis value counts:\n", df["diagnosis"].value_counts())

# Diagnosis distribution visualization
sns.countplot(x="diagnosis", data=df)
plt.title("Diagnosis Distribution (M=Malignant, B=Benign)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))

numeric_df = df.select_dtypes(include=["float64", "int64"])

corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Example feature distribution (radius_mean)
sns.histplot(data=df, x="feature1", hue="diagnosis", kde=True)
plt.title("Feature1 Distribution by Diagnosis")
plt.show()

# Train/validation split
X = df.drop(columns=["diagnosis"])  # Features
y = df["diagnosis"].map({"M": 1, "B": 0})  # Convert M=1, B=0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data size:", X_train.shape)
print("Validation data size:", X_val.shape)
