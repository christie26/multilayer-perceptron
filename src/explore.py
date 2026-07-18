import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Column names
columns = ["id", "diagnosis"] + [f"feature{i}" for i in range(1, 31)]


def main():
    parser = argparse.ArgumentParser(description="Explore the dataset")
    parser.add_argument("--data", type=str, default="data.csv", help="Input CSV file")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Validation split ratio"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data, names=columns)

    # Basic info
    print(f"Data size: {df.shape}")
#   df.info()
#   df.describe()

    # Check diagnosis value distribution
    print("\nDiagnosis value counts:\n", df["diagnosis"].value_counts())

    # Diagnosis distribution visualization
    sns.countplot(x="diagnosis", data=df)
    plt.title("Diagnosis Distribution (M=Malignant, B=Benign)")

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")

    # Example feature distribution (radius_mean)
    plt.figure()
    sns.histplot(data=df, x="feature1", hue="diagnosis", kde=True)
    plt.title("Feature1 Distribution by Diagnosis")
    plt.show()


if __name__ == "__main__":
    main()
