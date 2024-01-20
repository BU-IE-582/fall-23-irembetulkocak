#!/usr/bin/env python
# coding: utf-8

# # Generating the data 
# 
# With the parameters mean, median, standard deviation and pairwise covariance. 
# 

# In[3]:


import numpy as np
import pandas as pd

num_instances = 500
num_variables = 8


# Define parameters
# Mean, std, and covariance matrix (assuming covariance matrix is symmetric and positive definite)
mean = np.random.rand(num_variables)
std_dev = np.abs(np.random.rand(num_variables))  # std dev must be positive
cov_matrix = np.random.rand(num_variables, num_variables)
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Making the matrix symmetric
cov_matrix += num_variables * np.identity(num_variables)  # Making the matrix positive definite
data = np.random.multivariate_normal(mean, cov_matrix, size=num_instances)

# Create a DataFrame
df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(1, num_variables+1)])

#I choose the fourth parameter as median
desired_median = np.array([0.5] * num_variables)
max_iterations = 100
tolerance = 0.01  # Tolerance for the median difference
for _ in range(max_iterations):
    current_median = df.median().values
    median_diff = desired_median - current_median
    if np.all(np.abs(median_diff) <= tolerance):
        break
    df += median_diff

print(df)


# We will create Random Forest Similarity Matrix. 
# 
# It is stated that "Random forests can provide measures of the similarity between pairs of examples in the dataset. Each of the N examples is represented by a feature vector, all of which are passed down each tree in the forest. The similarities are initialised to zero, and if examples i and j finish in the same terminal node of a tree, their similarity sij is increased by one. The final pairwise similarity measures are normalised by the total number of trees in the forest."
# 
# We first train the model and create the similarity index.

# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = le.fit_transform(df.index)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df, labels)

similarity_matrix = np.zeros((num_instances, num_instances))

# Extracting the leaf indices for each instance in each tree
for tree in rf.estimators_:
    leaf_indices = tree.apply(df)
    
    # Increment the similarity matrix each time two instances share the same leaf
    for i in range(num_instances):
        for j in range(num_instances):
            if leaf_indices[i] == leaf_indices[j]:
                similarity_matrix[i, j] += 1

# Normalize by the number of trees to get a measure between 0 and 1
similarity_matrix /= rf.n_estimators

similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# Display the first few rows of the similarity matrix
print(similarity_df)


# In[9]:


# Transform RF similarity to dissimilarity by extracting it from 1
dissimilarity_matrix = 1 - similarity_df
dissimilarity_df = pd.DataFrame(dissimilarity_matrix, index=range(500), columns=range(500))

print(dissimilarity_df)


# In[10]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn_extra.cluster import KMedoids #I uploaded sckit_learn_extra
num_clusters = 4 #as stated in the homework


# In[16]:


# partitioning around medoids clustering algorithm
kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
kmedoids_labels = kmedoids.fit_predict(dissimilarity_matrix)

from sklearn.metrics.pairwise import euclidean_distances #since ward's method need euclidean distances

# Convert the dissimilarity matrix to a data set through MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_transformed = mds.fit_transform(dissimilarity_df)

# hierarchical clustering using Ward's method
ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
ward_labels = ward.fit_predict(mds_transformed)


df_clustered = df.copy()
df_clustered['KMedoids_Cluster'] = kmedoids_labels
df_clustered['Hierarchical_Cluster'] = ward_labels #Hierarchial Ward Method
df_clustered


# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# we use raw data to clsuter with kmeans, so we need to scale it first.
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df)
num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data_normalized)

# Combine the data and labels into a df
data_clustered_df = pd.DataFrame(data_normalized, columns=[f'var_{i}' for i in range(1, 9)])
data_clustered_df['KMeans_Cluster'] = kmeans_labels

data_clustered_df.head()


# In[21]:


# show three of them in a single df
df_clustered = df.copy()
df_clustered['KMeans_Cluster'] = kmeans_labels
df_clustered['Hierarchical_Cluster'] = ward_labels #Hierarchial Ward Method
df_clustered['KMedoids_Cluster'] = kmedoids_labels 
df_clustered


# ##  Mean Factor and Covariance Matrix.
# 
# 1. We compare the mean factors for each variable for each method.
# 
# 2. We compare the covariance matrix for each cluster for each method.
# 

# In[24]:


mean_factors_kmeans = {}
cov_matrices_kmeans = {}

# Calculate mean factor and covariance matrix for each cluster in KMeans
for cluster in range(num_clusters):
    cluster_data = df_clustered[df_clustered['KMeans_Cluster'] == cluster].drop(['KMeans_Cluster', 'Hierarchical_Cluster', 'KMedoids_Cluster'], axis=1)
    mean_factors_kmeans[cluster] = cluster_data.mean()
    cov_matrices_kmeans[cluster] = cluster_data.cov()

(mean_factors_kmeans, cov_matrices_kmeans)


# In[25]:


mean_factors_hierarchical = {}
cov_matrices_hierarchical = {}
# Calculate mean factor and covariance matrix for each cluster in Hierarchical clusterinf
for cluster in range(num_clusters):
    cluster_data = df_clustered[df_clustered['Hierarchical_Cluster'] == cluster].drop(['KMeans_Cluster', 'Hierarchical_Cluster', 'KMedoids_Cluster'], axis=1)
    mean_factors_hierarchical[cluster] = cluster_data.mean()
    cov_matrices_hierarchical[cluster] = cluster_data.cov()
(mean_factors_hierarchical, cov_matrices_hierarchical)


# In[26]:


mean_factors_kmedoids = {}
cov_matrices_kmedoids = {}

# Calculate mean factor and covariance matrix for each cluster in KMedoids
for cluster in range(num_clusters):
    cluster_data = df_clustered[df_clustered['KMedoids_Cluster'] == cluster].drop(['KMeans_Cluster', 'Hierarchical_Cluster', 'KMedoids_Cluster'], axis=1)
    mean_factors_kmedoids[cluster] = cluster_data.mean()
    cov_matrices_kmedoids[cluster] = cluster_data.cov()
    
(mean_factors_kmedoids, cov_matrices_kmedoids)


# In[34]:


from sklearn.metrics import silhouette_score


silhouette_kmeans = silhouette_score(data_normalized, kmeans_labels)
silhouette_hierarchical = silhouette_score(similarity_df, ward_labels)
silhouette_kmedoids = silhouette_score(similarity_df, kmedoids_labels)


print(f"Clustering Metrics:")
print(f"Silhouette Kmeans Score: {silhouette_kmeans}")
print(f"Silhouette Hierarchial Score: {silhouette_hierarchical}")
print(f"Silhouette Kmeadoids Score: {silhouette_kmedoids}")



# When we look at the Silhouette scores of each method, we see that Ward's method for hiererchial clustering gives negative score and the other two has values near 0. 
# 
# KMeans has the highest Silhouette score (0.093), suggesting that it has performed relatively better in creating distinct clusters than the other methods. However, the score is still close to zero, indicating that the clusters are not well-separated or are very dense. Hierarchical clustering with Ward's method has a negative Silhouette score (-0.0008), which implies overlapping clusters or incorrect cluster assignments. KMedoids has a score (0.0004) near zero, indicating overlapping clusters or an ineffective clustering structure.
# 
# 
# So these results tell us something about Random Forest Similarity as clustering data, but I will look at the other measures since silhoutte scores work best with convex shaped clusters.

# In[33]:


from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

calinski_harabasz_kmeans = calinski_harabasz_score(data_normalized, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(data_normalized, kmeans_labels)
calinski_harabasz_hierarchical = calinski_harabasz_score(similarity_df, ward_labels)
davies_bouldin_hierarchical = davies_bouldin_score(similarity_df, ward_labels)
calinski_harabasz_kmedoids = calinski_harabasz_score(similarity_df, kmedoids_labels)
davies_bouldin_kmedoids = davies_bouldin_score(similarity_df, kmedoids_labels)

print(f"Clustering Metrics:")
print(f"Calinski_harabasz Kmeans Score: {calinski_harabasz_kmeans}")
print(f"Calinski_harabasz Hierarchial Score: {calinski_harabasz_hierarchical}")
print(f"Calinski_harabasz Kmeadoids Score: {calinski_harabasz_kmedoids}")
print(f"")
print(f"Davies_bouldin Kmeans Score: {davies_bouldin_kmeans}")
print(f"Davies_bouldin Hierarchial Score: {davies_bouldin_hierarchical}")
print(f"Davies_bouldin Kmeadoids Score: {davies_bouldin_kmedoids}")




# Higher scores in the Calinski-Harabasz measurement are more appropriate for qualified clustering methods, and lower scores in the Davies-Bouldin index are indicative of well-defined clusters.
# 
# KMeans has the highest score (49.005) in the Calinski-Harabasz index, indicating that it has the best-defined clusters according to this metric. Hierarchical clustering and KMedoids have significantly lower scores, suggesting less distinct clusters.
# 
# KMeans also has the lowest score in the Davies-Bouldin index, which is desirable and suggests better clustering. Hierarchical clustering and KMedoids have high scores, implying that the clusters are not well-separated.
# 
# Overall, KMeans on the raw, scaled data appears to be the superior choice for this dataset based on these metrics. The relatively low scores in the Silhouette measurement across all methods suggest that the dataset may not have well-defined, convex clusters.
# 
# As a result, we can say that KMeans is the superior choice for clustering this data, or we might conclude that using scaled raw data yields better results than using the RF similarity matrix. The latter option is supported by the metrics, and it is known from the literature that KMeans does not inherently have a marked superiority over other methods.
# 
# The lower scores for RF similarity-based clustering could be due to the Random Forest Similarity preserving local neighborhood structures but not capturing the global structure needed for distinct clustering. This is especially likely if the clusters are not well-separated or if the cluster shapes are not well-suited to the metrics used for evaluation. Additionally, the performance of clustering algorithms can be highly dependent on the nature of the dataset and the underlying true structure of the data. If the data doesn't inherently have a clear cluster structure, the clustering metrics may not show strong results.
# 
# 
# ## The curse of dimensionality
# 
# If we increase the number of variables, it could decrease the likelihood that these methods will cluster the data points into separable groups. This is because, with higher dimensionality, points will exhibit distinctive characteristics, tending to form clusters of individual points. When attempting to merge these single-point clusters, the method may become confused, leading to a mix-up of clusters. As a result, defining successful and well-separated clusters becomes increasingly challenging, almost to the point of impossibility, following a logarithmic trend.
