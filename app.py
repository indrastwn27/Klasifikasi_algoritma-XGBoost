import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Cuaca Kolkata - XGBoost",
    page_icon="🌧️",
    layout="wide"
)

# Load model, scaler, label encoder, dan data evaluasi
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("le_season.pkl", "rb") as f:
        le_season = pickle.load(f)
    with open("le_wind.pkl", "rb") as f:
        le_wind = pickle.load(f)
    with open("eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)
    return model, scaler, le_season, le_wind, eval_data

try:
    model, scaler, le_season, le_wind, eval_data = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# 1. Judul dan Deskripsi Proyek
st.title("🌧️ Dashboard Klasifikasi Cuaca & Prediksi Hujan Kota Kolkata")
st.markdown(
    """
    Aplikasi ini menganalisis data cuaca harian Kota Kolkata (**Juni 2021 – Juni 2026**) dan memprediksi 
    **peluang terjadinya hujan (`is_rainy`)** berdasarkan parameter temperatur, kecepatan angin, durasi sinar matahari, 
    dan musim menggunakan algoritma **XGBoost Classifier** (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`).
    """
)

if not artifacts_loaded:
    st.error(
        "File artefak (`model.pkl`, `scaler.pkl`, `le_season.pkl`, `le_wind.pkl`, `eval_data.pkl`) "
        "tidak ditemukan. Jalankan dulu Notebook 'Klasifikasi_Cuaca_Kolkata.ipynb' sampai selesai "
        "supaya file-file tersebut dibuat, lalu taruh di folder yang sama dengan `app.py`."
    )
    st.stop()

data = eval_data['data']
metrics_df = eval_data['metrics_df']
cm = eval_data['cm']
fpr = eval_data['fpr']
tpr = eval_data['tpr']
auc_score = eval_data['auc']
importances = eval_data['importances']

# Navigasi Tab Dashboard
tab1, tab2, tab3 = st.tabs([
    "📊 Ringkasan Dataset & Grafik EDA",
    "📈 Evaluasi & Visualisasi Model XGBoost",
    "🔮 Form Input Prediksi Manual"
])

# ==========================================
# TAB 1: RINGKASAN DATASET & GRAFIK EDA
# ==========================================
with tab1:
    st.subheader("Ringkasan Dataset (`kolkata_weather_jan_june_2026.csv`)")
    total_rows = len(data)
    rain_count = int((data['is_rainy'] == 1).sum())
    no_rain_count = int((data['is_rainy'] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Pengamatan", f"{total_rows} Hari")
    c2.metric("Jumlah Fitur Prediktor (X)", "9 Fitur")
    c3.metric("Proporsi Kelas Hujan (1)", f"{rain_count} Hari ({rain_count/total_rows*100:.2f}%)")
    c4.metric("Proporsi Tidak Hujan (0)", f"{no_rain_count} Hari ({no_rain_count/total_rows*100:.2f}%)")

    with st.expander("Lihat Tabel Data Mentah & Ringkasan Statistik"):
        st.dataframe(data.head(10), use_container_width=True)
        st.dataframe(data.describe(), use_container_width=True)

    st.divider()
    st.subheader("Grafik Eksplorasi Data (EDA)")

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown("**1. Proporsi Kelas Target (`is_rainy`) & Distribusi Musim (`season`)**")
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(data=data, x='is_rainy', palette=['#3498db', '#e74c3c'], ax=ax1[0])
        ax1[0].set_xticklabels(['Tidak Hujan (0)', 'Hujan (1)'])
        ax1[0].set_title("Proporsi Kelas Target")

        sns.countplot(data=data, x='season', hue='is_rainy', palette='Set2', ax=ax1[1])
        ax1[1].set_title("Status Hujan per Musim")
        ax1[1].tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_eda2:
        st.markdown("**2. Distribusi Fitur Temperatur & Durasi Sinar Matahari**")
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data['temperature_2m_mean'], bins=25, kde=True, color='coral', ax=ax2[0])
        ax2[0].set_title("Distribusi Suhu Rata-Rata (°C)")

        sns.histplot(data['sunshine_duration'], bins=25, kde=True, color='gold', ax=ax2[1])
        ax2[1].set_title("Distribusi Durasi Sinar Matahari (Detik)")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("**3. Heatmap Korelasi Antar Fitur Numerik**")
    num_cols = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
                'precipitation_sum', 'rain_sum', 'precipitation_hours',
                'wind_speed_10m_max', 'wind_gusts_10m_max',
                'wind_direction_10m_dominant', 'sunshine_duration', 'is_rainy']
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.heatmap(data[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

# ==========================================
# TAB 2: EVALUASI & VISUALISASI MODEL
# ==========================================
with tab2:
    st.subheader("Performa Model XGBoost Classifier pada Data Uji (20%)")
    
    # Metrik Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics_df.loc[0, 'Nilai']*100:.2f}%")
    m2.metric("Precision", f"{metrics_df.loc[1, 'Nilai']*100:.2f}%")
    m3.metric("Recall", f"{metrics_df.loc[2, 'Nilai']*100:.2f}%")
    m4.metric("F1-Score", f"{metrics_df.loc[3, 'Nilai']*100:.2f}%")
    m5.metric("ROC-AUC Score", f"{metrics_df.loc[4, 'Nilai']:.4f}")

    st.markdown("**Tabel Ringkasan Metrik Evaluasi:**")
    st.dataframe(metrics_df.style.format({"Nilai": "{:.4f}"}), use_container_width=True)

    st.divider()
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tidak Hujan (0)', 'Hujan (1)'],
                    yticklabels=['Tidak Hujan (0)', 'Hujan (1)'], ax=ax_cm)
        ax_cm.set_xlabel("Prediksi Model")
        ax_cm.set_ylabel("Kelas Aktual")
        st.pyplot(fig_cm)

    with col_vis2:
        st.markdown("**Grafik ROC Curve**")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {auc_score:.3f})')
        ax_roc.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.divider()
    st.markdown("**Visualisasi Feature Importance (XGBoost)**")
    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis', ax=ax_fi)
    ax_fi.set_xlabel("Importance Score")
    ax_fi.set_ylabel("Fitur")
    st.pyplot(fig_fi)

# ==========================================
# TAB 3: FORM INPUT PREDIKSI MANUAL
# ==========================================
with tab3:
    st.subheader("Masukkan Parameter Cuaca Harian")

    col1, col2 = st.columns(2)

    with col1:
        temp_mean = st.slider("Suhu Rata-Rata Harian / temperature_2m_mean (°C)", 10.0, 40.0, 27.5, 0.1)
        temp_max = st.slider("Suhu Maksimum / temperature_2m_max (°C)", 15.0, 45.0, 31.5, 0.1)
        temp_min = st.slider("Suhu Minimum / temperature_2m_min (°C)", 8.0, 35.0, 24.5, 0.1)
        sunshine = st.number_input(
            "Durasi Sinar Matahari / sunshine_duration (Detik)",
            min_value=0.0, max_value=45000.0, value=35000.0, step=500.0,
            help="0 detik = mendung penuh seharian, ~36.000+ detik = cerah terik"
        )

    with col2:
        wind_speed = st.number_input("Kecepatan Angin Maks / wind_speed_10m_max (km/h)", 0.0, 60.0, 13.5, 0.5)
        wind_gusts = st.number_input("Hembusan Angin Maks / wind_gusts_10m_max (km/h)", 0.0, 100.0, 30.0, 0.5)
        wind_dir = st.slider("Arah Angin Dominan / wind_direction_10m_dominant (°)", 0, 360, 190)
        season = st.selectbox(
            "Musim (Season)",
            options=["Monsoon", "Post-Monsoon", "Summer", "Winter"],
            help="Monsoon (Jun-Sep), Post-Monsoon (Okt-Nov), Winter (Des-Feb), Summer (Mar-Mei)"
        )

    # Menentukan arah mata angin otomatis dari derajat angin sebagaimana di Notebook
    if wind_dir >= 315 or wind_dir < 45:
        wind_compass = "North"
    elif wind_dir >= 45 and wind_dir < 135:
        wind_compass = "East"
    elif wind_dir >= 135 and wind_dir < 225:
        wind_compass = "South"
    else:
        wind_compass = "West"

    st.caption(f"🧭 Kategori Arah Mata Angin Otomatis (`wind_compass`): **{wind_compass}**")
    st.divider()

    if st.button("🔍 Prediksi Cuaca Sekarang", type="primary", use_container_width=True):
        # Encode input kategorikal sesuai encoder saat training
        season_encoded = le_season.transform([season])[0]
        wind_encoded = le_wind.transform([wind_compass])[0]

        # Susun fitur sesuai urutan kolom saat training
        input_data = pd.DataFrame([{
            'temperature_2m_mean': temp_mean,
            'temperature_2m_max': temp_max,
            'temperature_2m_min': temp_min,
            'wind_speed_10m_max': wind_speed,
            'wind_gusts_10m_max': wind_gusts,
            'wind_direction_10m_dominant': wind_dir,
            'sunshine_duration': sunshine,
            'season': season_encoded,
            'wind_compass': wind_encoded
        }])

        # Scaling fitur
        input_scaled = scaler.transform(input_data)

        # Prediksi kelas dan probabilitas
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.subheader("Hasil Prediksi Model XGBoost")
        if prediction == 1:
            st.warning(f"🌧️ **BERPOTENSI HUJAN (Kelas 1)** — Probabilitas Hujan: **{probability[1]*100:.2f}%**")
        else:
            st.success(f"☀️ **TIDAK HUJAN / CERAH BERAWAN (Kelas 0)** — Probabilitas Tidak Hujan: **{probability[0]*100:.2f}%**")

        # Grafik Probabilitas
        prob_df = pd.DataFrame({
            "Status Cuaca": ["Tidak Hujan (0)", "Hujan (1)"],
            "Probabilitas (%)": [probability[0] * 100, probability[1] * 100]
        })
        st.bar_chart(prob_df.set_index("Status Cuaca"))

        with st.expander("Lihat Data Input Ter-encode yang Masuk ke Model"):
            st.dataframe(input_data)

st.divider()
st.caption("Model: XGBoost Classifier ● Dataset: Kolkata Weather (1.827 Hari) ● Dibuat dengan Streamlit")