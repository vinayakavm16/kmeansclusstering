import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Check for missing values
data.isnull().sum()

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Plot the distribution of Age, Annual Income (k$), and Spending Score
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['Annual Income (k$)'], kde=True, bins=30)
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data['Spending Score (1-100)'], kde=True, bins=30)
plt.title('Spending Score Distribution')

plt.show()

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_scaled)

# Add the cluster column to the original data
data['Cluster'] = clusters

# Silhouette Score
score = silhouette_score(data_scaled, clusters)
print(f'Silhouette Score: {score}')

# 2D Visualization
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set1')

# Customize x-axis tick labels to show ₹ symbol
scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{int(x)}k'))

plt.title('Customer Segments (2D)')
plt.xlabel('Annual Income (k₹)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='Set1')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k₹)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('Customer Segments (3D)')
plt.show()

# Count the number of customers in each category for each cluster
category_counts = data.groupby(['Cluster', 'Category']).size().reset_index(name='Count')

# Plotting the counts
plt.figure(figsize=(12, 8))
sns.barplot(x='Category', y='Count', hue='Cluster', data=category_counts, palette='Set1')
plt.title('Number of Customers in Each Category by Cluster')
plt.xlabel('Category')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(title='Cluster')
plt.show()

# Calculate average Annual Income and Spending Score for each cluster
cluster_summary = data.groupby('Cluster').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).reset_index()

# Visualize the cluster summary
plt.figure(figsize=(12, 6))
sns.barplot(x='Cluster', y='Annual Income (k$)', data=cluster_summary, palette='Set1')
plt.title('Average Annual Income by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Annual Income (k₹)')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Cluster', y='Spending Score (1-100)', data=cluster_summary, palette='Set1')
plt.title('Average Spending Score by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending Score (1-100)')
plt.show()

# Customer Lifetime Value (CLV) Prediction (Simplified Example)
# CLV = Average Annual Income * Average Spending Score * 0.1 (Assumption)
cluster_summary['CLV'] = cluster_summary['Annual Income (k$)'] * cluster_summary['Spending Score (1-100)'] * 0.1

# Visualize CLV by cluster
plt.figure(figsize=(12, 6))
sns.barplot(x='Cluster', y='CLV', data=cluster_summary, palette='Set1')
plt.title('Customer Lifetime Value (CLV) by Cluster')
plt.xlabel('Cluster')
plt.ylabel('CLV (₹)')
plt.show()

# Churn Prediction (Assumption: High spending score and low income might indicate potential churn)
# This is a hypothetical example and should be based on actual churn data
data['Potential Churn'] = (data['Spending Score (1-100)'] > 75) & (data['Annual Income (k$)'] < 40)
churn_counts = data.groupby(['Cluster', 'Potential Churn']).size().reset_index(name='Count')

# Visualize potential churn by cluster
plt.figure(figsize=(12, 8))
sns.barplot(x='Cluster', y='Count', hue='Potential Churn', data=churn_counts, palette='Set1')
plt.title('Potential Churn by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.legend(title='Potential Churn')
plt.show()

# Example of personalized marketing campaigns (hypothetical)
data['Campaign'] = data['Cluster'].map({
    0: 'Discount on Luxury Items',
    1: 'Exclusive Membership Offers',
    2: 'Seasonal Sale Promotions',
    3: 'Loyalty Rewards Program',
    4: 'Personalized Product Recommendations'
})

# Visualize campaigns by cluster
campaign_counts = data.groupby(['Cluster', 'Campaign']).size().reset_index(name='Count')

plt.figure(figsize=(12, 8))
sns.barplot(x='Campaign', y='Count', hue='Cluster', data=campaign_counts, palette='Set1')
plt.title('Marketing Campaigns by Cluster')
plt.xlabel('Campaign')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(title='Cluster')
plt.show()
