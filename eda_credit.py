# eda_credit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

# Create output folder
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('cs-training.csv', index_col=0)  # first column is Unnamed: 0 (ID)
print("Shape:", df.shape)
print(df.head())

# Rename columns for clarity (optional)
df.columns = df.columns.str.strip()

# Check target distribution
print("\nTarget distribution (SeriousDlqin2yrs):")
print(df['SeriousDlqin2yrs'].value_counts(normalize=True))

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing MonthlyIncome (median imputation)
imputer = SimpleImputer(strategy='median')
df['MonthlyIncome'] = imputer.fit_transform(df[['MonthlyIncome']])

# Also NumberOfDependents (mode imputation)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0], inplace=True)

# 1. Univariate distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
num_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfDependents']
for i, col in enumerate(num_cols):
    row, col_idx = divmod(i, 3)
    sns.histplot(df[col], kde=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(col)
plt.tight_layout()
plt.savefig('outputs/univariate.png')
plt.close()

# 2. Target vs continuous features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(num_cols):
    row, col_idx = divmod(i, 3)
    sns.boxplot(x=df['SeriousDlqin2yrs'], y=df[col], ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} vs SeriousDlqin2yrs')
plt.tight_layout()
plt.savefig('outputs/target_vs_features.png')
plt.close()

# 3. Correlation matrix (numeric only)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('outputs/correlation.png')
plt.close()

# 4. Key risk factors (top correlations with target)
target_corr = corr['SeriousDlqin2yrs'].sort_values(ascending=False)
print("\nTop features correlated with SeriousDlqin2yrs:")
print(target_corr)

# 5. Binning age to see default rate
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                          labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])
age_default = df.groupby('age_group')['SeriousDlqin2yrs'].mean()
plt.figure(figsize=(8,5))
age_default.plot(kind='bar')
plt.title('Default Rate by Age Group')
plt.ylabel('Proportion of SeriousDlqin2yrs')
plt.savefig('outputs/default_by_age.png')
plt.close()

# 6. RevolvingUtilizationOfUnsecuredLines – high utilization risk
df['high_util'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 1).astype(int)
util_default = df.groupby('high_util')['SeriousDlqin2yrs'].mean()
print("\nDefault rate when utilization > 1:", util_default[1])

# 7. Save a summary report
with open('outputs/insights.txt', 'w') as f:
    f.write("Credit Scoring EDA Summary\n")
    f.write("==========================\n\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Overall default rate: {df['SeriousDlqin2yrs'].mean():.4f}\n\n")
    f.write("Top 5 risk factors (correlation with target):\n")
    for feature, corr_val in target_corr.head(6).items():
        if feature != 'SeriousDlqin2yrs':
            f.write(f"  {feature}: {corr_val:.4f}\n")
    f.write("\nKey insights:\n")
    f.write("- RevolvingUtilizationOfUnsecuredLines > 1 is a strong warning sign.\n")
    f.write("- Younger and older age groups have higher default rates.\n")
    f.write("- DebtRatio alone is weakly correlated; but combined with other factors matters.\n")
    f.write("- NumberOfTime30-59DaysPastDueNotWorse is a powerful predictor.\n")

print("\n✅ EDA completed. Outputs saved in 'outputs/' folder.")
