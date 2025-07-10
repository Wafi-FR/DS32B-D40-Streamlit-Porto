import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("flight.csv")
    return df

# Data Preprocessing
def preprocess_data(df):
    columns_to_drop = [
        'MEMBER_NO', 'FFP_DATE', 'FIRST_FLIGHT_DATE',
        'LOAD_TIME', 'LAST_FLIGHT_DATE', 'WORK_CITY',
        'WORK_PROVINCE', 'WORK_COUNTRY', 'GENDER'
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

# Clustering
def perform_clustering(df_cleaned, n_clusters=4):
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df_cleaned)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(data_std)
    df_cleaned['clusters'] = kmeans.labels_
    data_std_df = pd.DataFrame(data=data_std, columns=df_cleaned.columns[:-1])
    data_std_df['clusters'] = kmeans.labels_
    return df_cleaned, data_std_df, data_std

# Find optimal k
def find_optimal_k(data_std):
    silhouette_scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_std)
        score = silhouette_score(data_std, labels)
        silhouette_scores[k] = score
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    return optimal_k, silhouette_scores

# Main App
def main():
    st.set_page_config(layout="wide")
    st.title("✈️ Customer Segmentation Analysis (Flight Data)")

    st.write("")

    df = load_data()
    st.subheader("Original Data Preview")
    st.write(df.head())

    st.subheader("Data Preprocessing")
    df_cleaned = preprocess_data(df.copy())
    st.write(f"Shape after preprocessing: {df_cleaned.shape}")
    st.dataframe(df_cleaned.head())

    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Statistical Summary")
    st.write(df_cleaned.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # Distribution plots
    st.subheader("Distribution of Numeric Features")
    num_cols = df_cleaned.columns
    n_cols_dist = 3
    n_rows_dist = (len(num_cols) + n_cols_dist - 1) // n_cols_dist
    fig_dist, axes = plt.subplots(n_rows_dist, n_cols_dist, figsize=(18, n_rows_dist * 4))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df_cleaned[col], bins=30, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f'Distribution of {col}')

    for j in range(i + 1, len(axes)):
        fig_dist.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig_dist)

    # Clustering
    st.subheader("K-Means Clustering")
    df_clustered, data_std_df, data_std = perform_clustering(df_cleaned.copy())

    st.write("Data with cluster assignments")
    st.write(df_clustered.head())

    # Elbow Method
    st.subheader("Elbow Method for Optimal k")
    inertia = []
    range_k = range(2, 11)
    for i in range_k:
        km = KMeans(n_clusters=i, random_state=0, n_init=10)
        km.fit(data_std)
        inertia.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=list(range_k), y=inertia, ax=ax_elbow)
    sns.scatterplot(x=list(range_k), y=inertia, s=200, color='red', ax=ax_elbow)
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("Number of Clusters")
    ax_elbow.set_ylabel("Inertia")
    st.pyplot(fig_elbow)

    # Silhouette Score
    st.subheader("Silhouette Score Analysis")
    optimal_k, silhouette_scores = find_optimal_k(data_std)

    fig_silhouette, ax_silhouette = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), marker='o', ax=ax_silhouette)
    ax_silhouette.set_title("Silhouette Scores")
    st.pyplot(fig_silhouette)

    st.success(f"Optimal k = {optimal_k} with Silhouette Score: {silhouette_scores[optimal_k]:.4f}")

    if df_clustered['clusters'].nunique() != optimal_k:
        st.info(f"Re-running clustering with optimal k = {optimal_k}")
        df_clustered, data_std_df, data_std = perform_clustering(df_cleaned.copy(), n_clusters=optimal_k)

    # PCA Cluster Visualization
    st.subheader("PCA Cluster Visualization")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data_std_df.drop('clusters', axis=1))
    data_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    data_pca['clusters'] = df_clustered['clusters']

    fig_pca, ax_pca = plt.subplots(figsize=(15, 10))
    sns.scatterplot(data=data_pca, x='PC1', y='PC2', hue='clusters', palette='viridis', s=100, ax=ax_pca)
    ax_pca.set_title("PCA Projection of Clusters")
    st.pyplot(fig_pca)

    # Cluster Profiling
    st.subheader("Cluster Profiling")
    st.write(df_clustered.groupby('clusters').agg(['mean', 'median']))

    st.subheader("Insights and Recommendations")
    st.write("""
    - **Cluster 0 (Dormant Loyal):** High FFP_TIER but low activity. Recommend re-engagement campaigns.
    - **Cluster 1 (Loyal Active):** High tier and frequent usage. Educate about point redemptions.
    - **Cluster 2 (High Value Elite):** Very active and valuable. Offer exclusive loyalty benefits.
    - **Cluster 3 (New/Passive):** Low activity. Encourage onboarding and promotions.
    """)

if __name__ == "__main__":
    main()
