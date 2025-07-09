
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("✈️ Clustering Analysis on Flight Data")

st.markdown("## Dataset: `flight.csv`")
df = pd.read_csv("flight.csv")
st.write("### Preview Data")
st.dataframe(df.head())

# Show missing values
st.write("### Missing Values")
st.write(df.isnull().sum())

# Scaling
features = df.select_dtypes(include=np.number).columns
df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)

# Select number of clusters
st.sidebar.header("K-Means Settings")
k = st.sidebar.slider("Select number of clusters (K):", min_value=2, max_value=10, value=3)

# KMeans Clustering
model = KMeans(n_clusters=k, random_state=42)
clusters = model.fit_predict(df_scaled)
df['cluster'] = clusters

# PCA for 2D visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(df_scaled)
df['PCA1'] = reduced[:,0]
df['PCA2'] = reduced[:,1]

# Plot PCA result
st.subheader("🧠 PCA Clustering Result")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='Set2', ax=ax)
plt.title("Flight Clusters (PCA)")
st.pyplot(fig)

# Cluster distribution
st.subheader("📊 Cluster Distribution")
st.bar_chart(df['cluster'].value_counts())

# Display summary stats per cluster
st.subheader("📈 Cluster Summary (Mean by Cluster)")
st.dataframe(df.groupby('cluster')[features].mean().round(2))
