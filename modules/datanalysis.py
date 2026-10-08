import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.ExcelFile('./data/rotterdam_dataset_37.xlsx')
df = pd.read_excel("./data/rotterdam_dataset_37.xlsx", sheet_name="Vessel_Calls")

df.info()

#null values, duplicates and descriptive statistics
print(df.isnull().sum())
print(df.duplicated().sum())    
print(df.describe(include='all').to_string())    

#Unique values for each column
for column in df.columns:
    unique_values = df[column].nunique()
    print(f"Column '{column}' has {unique_values} unique values.")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot for Cargo_tons
sns.boxplot(x=df['Cargo_tons'], ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Boxplot of Cargo Tons')

# Histogram for Cargo_tons
sns.histplot(df['Cargo_tons'], bins=50, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Distribution of Cargo Tons')

# Boxplot for Berth_Time_hr
sns.boxplot(x=df['Berth_Time_hr'], ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Boxplot of Berth Time (Hours)')

# Histogram for Berth_Time_hr
sns.histplot(df['Berth_Time_hr'], bins=50, kde=True, ax=axes[1, 1], color='lightgreen')
axes[1, 1].set_title('Distribution of Berth Time (Hours)')

plt.tight_layout()
plt.show()

# Calculate IQR for Cargo_tons to identify specific outliers
Q1_c = df['Cargo_tons'].quantile(0.25)
Q3_c = df['Cargo_tons'].quantile(0.75)
IQR_c = Q3_c - Q1_c
outliers_cargo = df[(df['Cargo_tons'] < (Q1_c - 1.5 * IQR_c)) | (df['Cargo_tons'] > (Q3_c + 1.5 * IQR_c))]

# Calculate IQR for Berth_Time_hr
Q1_b = df['Berth_Time_hr'].quantile(0.25)
Q3_b = df['Berth_Time_hr'].quantile(0.75)
IQR_b = Q3_b - Q1_b
outliers_berth = df[(df['Berth_Time_hr'] < (Q1_b - 1.5 * IQR_b)) | (df['Berth_Time_hr'] > (Q3_b + 1.5 * IQR_b))]

print(f"Number of outliers in Cargo_tons: {len(outliers_cargo)}")
print(f"Number of outliers in Berth_Time_hr: {len(outliers_berth)}")