import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Load CO2 emissions dataset
df = pd.read_csv('data.csv')  # Replace with actual file path
df_gdp = pd.read_csv('gdp_data.csv', skiprows=4)  # Skip metadata rows

# Display basic info
print(df.head())
print(df.info())

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Reshape GDP data
year_columns = [col for col in df_gdp.columns if col.isdigit()]  # Extract only year columns
df_gdp = df_gdp.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                      value_vars=year_columns, var_name='Date', value_name='GDP')

df_gdp.rename(columns={'Country Name': 'Country'}, inplace=True)
df_gdp['Date'] = pd.to_datetime(df_gdp['Date'], format='%Y', errors='coerce')  # Handle errors safely
df_gdp = df_gdp.dropna(subset=['Date', 'GDP'])  # Drop invalid rows

# Trend Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Kilotons of Co2', hue='Region', estimator='sum')
plt.title('Global CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions (Kilotons)')
plt.legend(title='Region')
plt.show()

# Top and Bottom CO2 Emission Per Capita
top_emitters = df.groupby('Country')['Metric Tons Per Capita'].mean().nlargest(10)
bottom_emitters = df.groupby('Country')['Metric Tons Per Capita'].mean().nsmallest(10)
print("Top 10 CO2 Emissions Per Capita:")
print(top_emitters)
print("Bottom 10 CO2 Emissions Per Capita:")
print(bottom_emitters)

# GDP vs CO2 Emissions
merged_df = df.merge(df_gdp, on=['Country', 'Date'], how='inner')

X = merged_df[['GDP']]
y = merged_df['Kilotons of Co2']
model = LinearRegression()
model.fit(X, y)

print(f'Linear Regression Coefficient (GDP vs CO2): {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=merged_df['GDP'], y=merged_df['Kilotons of Co2'], alpha=0.5)
sns.lineplot(x=merged_df['GDP'], y=model.predict(X), color='red')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions (Kilotons)')
plt.title('GDP vs CO2 Emissions')
plt.show()

# Clustering Countries Based on Emissions
X_cluster = df.groupby('Country')[['Kilotons of Co2']].mean()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster)
X_cluster['Cluster'] = kmeans.labels_
print(X_cluster.sort_values('Cluster'))

# --- Data Visualization Section ---

# Bar Chart: Top CO2 Emitters Per Capita
plt.figure(figsize=(10, 6))
sns.barplot(x=top_emitters.values, y=top_emitters.index, palette='Reds_r')
plt.title('Top 10 CO2 Emitters Per Capita')
plt.xlabel('Metric Tons Per Capita')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# Bar Chart: Bottom CO2 Emitters Per Capita
plt.figure(figsize=(10, 6))
sns.barplot(x=bottom_emitters.values, y=bottom_emitters.index, palette='Blues')
plt.title('Bottom 10 CO2 Emitters Per Capita')
plt.xlabel('Metric Tons Per Capita')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# Heatmap: Correlation Between Emissions, GDP and Per Capita Emissions
correlation_df = merged_df[['Kilotons of Co2', 'Metric Tons Per Capita', 'GDP']].dropna()
corr = correlation_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap: CO2, GDP, and Per Capita Emissions')
plt.tight_layout()
plt.show()

