import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Define file path (use relative path for portability)
file_path = "data/masculinity.csv"

# Load survey data
survey_data = pd.read_csv(file_path)

# Map string responses to numbers
strings_to_map = [
    "q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
    "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008",
    "q0007_0009", "q0007_0010", "q0007_0011"
]

# Convert categorical responses to numerical values
mapping = {
    "Often": 4,
    "Sometimes": 3,
    "Rarely": 2,
    "Never, but open to it": 1,
    "Never, and not open to it": 0
}

for string in strings_to_map:
    survey_data[string] = survey_data[string].map(mapping)

# Select relevant columns and remove rows with NaN values
relevant_data = [
    "q0007_0001", "q0007_0002", "q0007_0003",
    "q0007_0005", "q0007_0008", "q0007_0009"
]
rows_to_cluster = survey_data.dropna(subset=relevant_data)

# Perform K-Means clustering
classifier = KMeans(n_clusters=2, n_init=10)
classifier.fit(rows_to_cluster[relevant_data])

# Separate data into clusters
cluster_zero_indices = [i for i, label in enumerate(classifier.labels_) if label == 0]
cluster_one_indices = [i for i, label in enumerate(classifier.labels_) if label == 1]

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

# Print information about clusters' education and age
print("Cluster 0 - Education Distribution:")
print(cluster_zero_df['educ4'].value_counts())

print("Cluster 1 - Education Distribution:")
print(cluster_one_df['educ4'].value_counts())

print("Cluster 0 - Age Distribution:")
print(cluster_zero_df['age3'].value_counts())

print("Cluster 1 - Age Distribution:")
print(cluster_one_df['age3'].value_counts())
