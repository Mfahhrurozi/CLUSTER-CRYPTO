import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

    # --- Pilih fitur numerik secara otomatis ---
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Dataset tidak memiliki cukup fitur numerik untuk clustering.")
    else:
        # --- Imputasi nilai NaN ---
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df[numeric_cols])

        # --- Standarisasi ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # --- PCA ---
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # --- Clustering ---
        clusters = model.predict(X_pca)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]
        df['Cluster'] = clusters

        # --- Dashboard Header Cards ---
        col1, col2, col3 = st.columns(3)
        col1.metric("🧮 Total Data", len(df))
        col2.metric("🧩 Total Clusters", len(set(clusters)))
        top_cluster = df['Cluster'].value_counts().idxmax()
        top_count = df['Cluster'].value_counts().max()
        col3.metric("🔥 Dominant Cluster", f"Cluster {top_cluster}", f"{top_count} items")

        st.markdown("---")

        # --- Visualization Row ---
        col4, col5 = st.columns((2, 1))
        with col4:
            st.subheader("📍 PCA Cluster Plot")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, edgecolor='black', ax=ax1)
            ax1.set_title("Visualisasi Cluster Berdasarkan PCA")
            st.pyplot(fig1)

        with col5:
            st.subheader("📊 Jumlah Tiap Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2, palette='Set2')
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Jumlah")
            ax2.set_title("Distribusi Data per Cluster")
            st.pyplot(fig2)

        st.markdown("---")

        # --- Tabel Lengkap ---
        st.subheader("📋 Data Lengkap dengan Cluster")
        ordered_cols = ['PCA1', 'PCA2', 'Cluster'] + [col for col in df.columns if col not in ['PCA1', 'PCA2', 'Cluster']]
        st.dataframe(df[ordered_cols], use_container_width=True)

else:
    st.info("Silakan upload file CSV terlebih dahulu di sidebar.")
