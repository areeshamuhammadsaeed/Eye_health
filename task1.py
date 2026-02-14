import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────
# 1. Load & Clean the Dataset
# ────────────────────────────────────────────────
df = pd.read_csv('eye_score.csv')

# Drop unwanted column (leading comma / Unnamed: 0)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Rename columns for clarity and consistency
df = df.rename(columns={
    'id': 'ID',
    'exercise_hours': 'Exercise_Hours',
    'mental_health_score': 'Mental_Health_Score',
    'screen_time_hours': 'Screen_Time_Hours',
    'screen_brightness_avg': 'Screen_Brightness_Avg',
    'age': 'Age',
    'height_cm': 'Height_cm',
    'outdoor_light_exposure_hours': 'Outdoor_Light_Hours',
    'night_mode_usage': 'Night_Mode_Usage',
    'screen_distance_cm': 'Screen_Distance_cm',
    'glasses_number': 'Glasses_Number',
    'eye_health_score': 'Eye_Health_Score'
})

# Final checks
print("Shape after cleaning:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Basic info
print("\nDataset: 10,000 records on lifestyle, screen habits, demographics, and Eye Health Score (0–100).")
print("Target: Eye_Health_Score — higher = better eye health.\n")

# ────────────────────────────────────────────────
# 2. Visualizations (at least 5 different types)
# ────────────────────────────────────────────────

# 1. Histogram - Eye Health Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Eye_Health_Score'], bins=40, kde=True, color='teal')
plt.title('Distribution of Eye Health Score')
plt.xlabel('Eye Health Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('viz_01_eye_health_histogram.png', dpi=150)
plt.close()

# 2. Countplot (Bar-like) - Glasses Number distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Glasses_Number', data=df, hue='Glasses_Number', palette='viridis', legend=False)
plt.title('Count of Glasses Number')
plt.xlabel('Glasses Number')
plt.ylabel('Count')
plt.savefig('viz_02_glasses_count.png', dpi=150)
plt.close()

# 3. Bar Chart - Average Eye Health by Glasses Number
plt.figure(figsize=(8, 5))
sns.barplot(x='Glasses_Number', y='Eye_Health_Score', data=df, hue='Glasses_Number', errorbar=None, palette='magma', legend=False)
plt.title('Average Eye Health Score by Glasses Number')
plt.ylabel('Mean Eye Health Score')
plt.savefig('viz_03_eye_vs_glasses_bar.png', dpi=150)
plt.close()

# 4. Heatmap - Correlation matrix
corr = df.corr(numeric_only=True)
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.4)
plt.title('Correlation Heatmap - All Numerical Features')
plt.tight_layout()
plt.savefig('viz_04_correlation_heatmap.png', dpi=150)
plt.close()

# 5. Scatter Plots - Key relationships (3-in-1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.scatterplot(x='Screen_Time_Hours', y='Eye_Health_Score', data=df, alpha=0.35, color='coral', ax=axes[0])
axes[0].set_title('Screen Time vs Eye Health')

sns.scatterplot(x='Outdoor_Light_Hours', y='Eye_Health_Score', data=df, alpha=0.35, color='green', ax=axes[1])
axes[1].set_title('Outdoor Light vs Eye Health')

sns.scatterplot(x='Exercise_Hours', y='Eye_Health_Score', data=df, alpha=0.35, color='purple', ax=axes[2])
axes[2].set_title('Exercise Hours vs Eye Health')

plt.tight_layout()
plt.savefig('viz_05_key_scatter_plots.png', dpi=150)
plt.close()

# 6. Boxplot - Eye Health by Glasses Number
plt.figure(figsize=(8, 6))
sns.boxplot(x='Glasses_Number', y='Eye_Health_Score', data=df, hue='Glasses_Number', palette='Set2', legend=False)
plt.title('Eye Health Score Distribution by Glasses Number')
plt.savefig('viz_06_eye_health_boxplot.png', dpi=150)
plt.close()

# 7. Line Graph - Average Eye Health by Age Group (binned)
df['Age_Group'] = pd.cut(df['Age'], bins=10)
age_trend = df.groupby('Age_Group', observed=True)['Eye_Health_Score'].mean()

plt.figure(figsize=(10, 6))
age_trend.plot(kind='line', marker='o', color='darkblue')
plt.title('Average Eye Health Score by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Eye Health Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_07_age_trend_line.png', dpi=150)
plt.close()

# 3. Print Key Correlations for reference

print("\nCorrelations with Eye_Health_Score (sorted):")
print(corr['Eye_Health_Score'].sort_values(ascending=False).round(3))

print("\nAll visualizations saved successfully!")
print("Files created: viz_01 to viz_07 .png")