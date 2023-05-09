#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



#
# Read GDP data from CSV file
gdp_data = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5358352.csv', skiprows=4)

# Select columns of interest and rename them
data = gdp_data[['Country Name', '2019']]
data.columns = ['Country', 'GDP']


# Remove rows with missing values
data.dropna(inplace=True)

# Convert GDP column to float type
data['GDP'] = data['GDP'].astype(float)

# Compute summary statistics
mean_gdp = np.mean(data['GDP'])
median_gdp = np.median(data['GDP'])

# Transpose the data
data_transposed = data.transpose()
print(data_transposed)



# Sort data by GDP
top_gdp_countries = data.sort_values('GDP', ascending=False).head(10)

# Create a line chart for the top 10 countries by GDP
plt.figure(figsize=(8, 6))
sns.lineplot(data=top_gdp_countries, x='Country', y='GDP', marker='o')
plt.xticks(rotation=45)
plt.ylabel('GDP')
plt.title('Top 10 Countries by GDP in 2019')
plt.savefig('top_10_gdp_countries.png')
plt.show()


# Cluster the data using KMeans
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[['GDP']])
kmeans = KMeans(n_clusters=3, random_state=0).fit(normalized_data)

# Add cluster labels to the data
data['Cluster'] = kmeans.labels_

# Compute mean GDP for each cluster
cluster_means = data.groupby('Cluster')['GDP'].mean()

# Create a bar chart of mean GDP by cluster
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(cluster_means.index, cluster_means.values)
ax.set_xlabel('Cluster')
ax.set_ylabel('Mean GDP')
ax.set_title('Mean GDP by Cluster')
plt.show()


# Set plot style and colors
sns.set_style('whitegrid')
colors = sns.color_palette('husl', 8)
# Set plot style and colors
sns.set_style('whitegrid')
colors = sns.color_palette('husl', 8)

# Plot a box plot of GDP values
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='GDP', color=colors[1], width=0.5)
plt.xlabel('GDP')
plt.title('Distribution of GDP in 2019', fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Define the function for predicting GDP growth
def predict_gdp_growth(country, gdp, year, model):
    """
    Predicts the GDP growth for a given country, GDP, year, and model.
    """
    # Scale the input data
    scaler = StandardScaler()
    scaled_gdp = scaler.fit_transform([[gdp]])

    # Predict the GDP growth using the model
    predicted_growth = model.predict(scaled_gdp)[0]

    # Compute the new GDP
    new_gdp = gdp * (1 + predicted_growth)

    # Return the predicted GDP growth and new GDP
    return predicted_growth, new_gdp

# Test the prediction function
kmeans = KMeans(n_clusters=3, random_state=0)
normalized_data = scaler.fit_transform(data[['GDP']])
kmeans.fit(normalized_data)
test_country = 'United States'
test_gdp = 21.44e12
test_year = 2020
predicted_growth, new_gdp = predict_gdp_growth(test_country, test_gdp, test_year, kmeans)

print(f"The predicted GDP growth for {test_country} in {test_year} is {predicted_growth:.2f}.")
print(f"The new GDP for {test_country} in {test_year+1} is {new_gdp:.2f} trillion USD.")




# Define a list of years to predict GDP growth for
years = [2020, 2021, 2022, 2023, 2024, 2025]

# Predict GDP growth and new GDP for each year for United States
us_gdp_list = [test_gdp]
for year in years:
    _, new_gdp = predict_gdp_growth(test_country, us_gdp_list[-1], year, kmeans)
    us_gdp_list.append(new_gdp)

# Predict GDP growth and new GDP for each year for China
china_gdp = 14.14e12
china_gdp_list = [china_gdp]
for year in years:
    _, new_gdp = predict_gdp_growth('China', china_gdp_list[-1], year, kmeans)
    china_gdp_list.append(new_gdp)

# Plot the predicted GDP growth over the years for both countries
plt.plot(years, us_gdp_list[1:], '-o', label='United States')
plt.plot(years, china_gdp_list[1:], '-o', label='China')
plt.xlabel('Year')
plt.ylabel('GDP (trillion USD)')
plt.title('Predicted GDP Growth')
plt.legend()
plt.show()
# Plot the cluster graph
def plot_cluster_graph(normalized_data, kmeans):
    """
    Plot the cluster graph.
    """
    cluster_labels = kmeans.predict(normalized_data)
    silhouette_avg = silhouette_score(normalized_data, cluster_labels)
    
    plt.title(f"KMeans Clustering with Silhouette Score: {silhouette_avg}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Data")
    
    n = 30 # number of data points to plot
    for i in range(n):
        plt.scatter(i, normalized_data[i], c=cluster_labels[i], cmap='Reds_r', s=30)

    
    plt.show()       
    
plot_cluster_graph(normalized_data, kmeans)

# Merge the world shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_data = world.merge(data, left_on='name', right_on='Country')

# Create a map representation of GDP
fig, ax = plt.subplots(figsize=(8, 6))
world_data.plot(column='GDP', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('GDP by Country in 2019')
plt.savefig('gdp_map.png')
plt.show()
# Add poster-specific modifications
ax.tick_params(axis='both', labelsize=24)
ax.margins(0.1, 0.1)
fig.tight_layout()

# Save figure as poster.png
fig.savefig("poster.png", dpi=300)


# In[ ]:




