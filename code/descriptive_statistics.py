import os
import pandas as pd

# ==========================================================
# 1. Data Loading
# ==========================================================
print("=" * 70)
print("Descriptive Statistical Analysis")
print("=" * 70)

while True:
    base_path = input("\nEnter the full folder path containing data.csv:\n> ")
    base_path = base_path.strip().replace('"', '').replace("'", "")
    data_path = os.path.join(base_path, "data.csv")

    if os.path.exists(data_path):
        print(f"File found: {data_path}")
        break
    else:
        print("Error: data.csv not found. Please try again.")

data = pd.read_csv(data_path)
print("\nData successfully loaded.")
print(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")

# ==========================================================
# 2. Descriptive Statistics Table
# ==========================================================
print("\n" + "=" * 70)
print("1. Descriptive Statistics for Numerical Features")
print("=" * 70)

# Numerical columns
numerical_cols = ['Days', 'S/D', 'L/D', 'IF']

# Calculate descriptive statistics
desc_stats = data[numerical_cols].describe().T

# Add additional statistical measures
desc_stats['variance'] = data[numerical_cols].var()
desc_stats['skewness'] = data[numerical_cols].skew()
desc_stats['kurtosis'] = data[numerical_cols].kurtosis()
desc_stats['missing'] = data[numerical_cols].isnull().sum()

# Reorder columns for better readability
desc_stats = desc_stats[['count', 'mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis', 'missing']]

print("\nComplete Descriptive Statistics:")
print(desc_stats.to_string())

# ==========================================================
# 3. Save Output
# ==========================================================
stats_path = os.path.join(base_path, "Descriptive_Statistics.csv")
desc_stats.to_csv(stats_path)
print(f"\nDescriptive statistics table saved at:\n{stats_path}")

print("\n" + "=" * 70)
print("Descriptive statistical analysis completed successfully.")
print("=" * 70)
