import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
model = load("crypto_cluster_model.joblib")

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/50/000000/bitcoin.png", width=60)
    st.markdown("## Crypto Clustering")
    uploaded_file = st.file_uploader("📂 Upload Crypto CSV", type=["csv"])
    st.markdown("---")

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #262730;'>📊 Crypto Clustering Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = [
        'price_change_percentage_24h',
        'price_change_percentage_7d',
        'price_change_percentage_14d',
        'price_change_percentage_30d',
        'price_change_percentage_60d',
        'price_change_percentage_200d',
        'price_change_percentage_1y'
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    clusters = model.predict(X_pca)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['Cluster'] = clusters

    # --- Dashboard Header Cards ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🧮 Total Crypto", len(df))
    col2.metric("🧩 Total Clusters", len(set(clusters)))
    top_cluster = df['Cluster'].value_counts().idxmax()
    top_count = df['Cluster'].value_counts().max()
    col3.metric("🔥 Dominant Cluster", f"Cluster {top_cluster}", f"{top_count} items")

    st.markdown("---")

    # --- Visualization Row ---
    col4, col5 = st.columns((2, 1))
    with col4:
        st.subheader("📍 PCA Cluster Plot")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, edgecolor='black')
        st.pyplot(plt)

    with col5:
        st.subheader("📊 Jumlah Tiap Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2, palette='Set2')
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

    st.markdown("---")

    # --- Tabel Lengkap ---
    st.subheader("📋 Data Lengkap dengan Cluster")
    st.dataframe(df[['PCA1', 'PCA2', 'Cluster'] + ([col for col in df.columns if col not in ['PCA1', 'PCA2', 'Cluster']])], use_container_width=True)

else:
    st.info("Silakan upload file CSV terlebih dahulu di sidebar.")
