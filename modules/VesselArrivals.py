import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = pd.read_excel("./data/rotterdam_dataset_37.xlsx", sheet_name="Vessel_Calls")


# Vessel Arrivals rate
df.sort_values(by='Arrival_Date', inplace=True)
arrivals = df['Arrival_Date']

inter_arrival_times = np.diff(arrivals)
inter_arrival_hours = inter_arrival_times / np.timedelta64(1, 'h')

df['Inter_Arrival_Hours'] = df['Arrival_Date'].diff() / np.timedelta64(1, 'h')

#estimate the arrival rate (lambda) as the inverse of the mean inter-arrival time
arrival_rate = 1 / np.mean(inter_arrival_hours)

print(f"Mean inter-arrival time: {np.mean(inter_arrival_hours):.2f} hours")
print(f"Arrival rate (lambda): {arrival_rate:.4f} vessels per hour")
print(f"Daily arrival rate: {arrival_rate * 24:.2f} vessels per day")


#plot
plt.figure()
plt.hist(inter_arrival_hours, bins=50, rwidth=0.8, density=True)

# Add the theoretical exponential density line to compare[cite: 4]
fitExpDist = stats.expon(scale=1/arrival_rate)
xs = np.arange(0, np.max(inter_arrival_hours), 0.1)
plt.plot(xs, fitExpDist.pdf(xs), color='g', label=f'Exponential ($\lambda$={arrival_rate:.2f})')

plt.title('Distribution of Vessel Inter-arrival Times')
plt.xlabel('Inter-arrival Time (Hours)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Inter-arrival rate

mean_ia = np.mean(inter_arrival_hours)
var_ia = np.var(inter_arrival_hours)
std_ia = np.std(inter_arrival_hours)

print(f"Mean: {mean_ia}")
print(f"Variance: {var_ia}")
print(f"Standard Deviation: {std_ia}")


# Seasonal effects in these arrival statistics

df['Month'] = df['Arrival_Date'].dt.month
df['DayName'] = df['Arrival_Date'].dt.day_name()

monthly_stats = df.groupby('Month')['Arrival_Date'].agg(['count', 'mean', 'std']).reset_index()

#aggragetae day of week statistics
dayofweekorder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['DayName'] = pd.Categorical(df['DayName'], categories=dayofweekorder, ordered=True)
dow_stats = df.groupby('DayName')['Inter_Arrival_Hours'].agg(['mean', 'std', 'var', 'count']).reset_index()

print("Monthly Arrival Statistics:")
print(monthly_stats.to_string(index=False))

print("Day of Week Arrival Statistics:")
print(dow_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot by Month
sns.barplot(x='Month', y='mean', data=monthly_stats, ax=axes[0], color='skyblue', capsize=0.1)
axes[0].set_title('Average Inter-Arrival Time by Month')
axes[0].set_ylabel('Mean Hours')
axes[0].set_xlabel('Month (1=Jan, 12=Dec)')

# Plot by Day of Week
sns.barplot(x='DayName', y='mean', data=dow_stats, ax=axes[1], color='lightgreen', capsize=0.1)
axes[1].set_title('Average Inter-Arrival Time by Day of Week')
axes[1].set_ylabel('Mean Hours')
axes[1].set_xlabel('Day of the Week')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Vessel categories differences in these arrival statistics

df['Arrival_by_Vessel_Category'] = df.groupby('Vessel_Type')['Arrival_Date'].diff() / np.timedelta64(1, 'h')
type_stats = df.groupby('Vessel_Type')['Arrival_by_Vessel_Category'].agg(['mean', 'std', 'var', 'count']).reset_index()

type_stats['Lambda (per hour)'] = 1 / type_stats['mean']
type_stats['Daily Arrival Rate'] = type_stats['Lambda (per hour)'] * 24

print("Arrival Statistics by Vessel Category:")
print(type_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot for variability
sns.boxplot(x='Vessel_Type', y='Arrival_by_Vessel_Category', data=df, ax=axes[0], palette='Set2')
# Cap the Y-axis at 20 hours to focus on the main distributions and ignore massive outliers
axes[0].set_ylim(0, 20) 
axes[0].set_title('Inter-Arrival Time Variability by Category')
axes[0].set_ylabel('Inter-Arrival Time (Hours)')

# Barplot for Daily Rate
sns.barplot(x='Vessel_Type', y='Daily Arrival Rate', data=type_stats, ax=axes[1], palette='Set2')
axes[1].set_title('Average Daily Arrivals by Category')
axes[1].set_ylabel('Vessels per Day')

plt.tight_layout()
plt.show()