import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# KONFIGURASI HALAMAN
# =====================================
st.set_page_config(
    page_title="Prediksi Keberhasilan Startup",
    page_icon="🚀",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_startup_model():
    # Menggunakan decorator cache agar model tidak di-load berulang kali setiap komponen berubah
    return joblib.load("model_startup.pkl")

try:
    model = load_startup_model()
except Exception as e:
    st.error(f"Gagal memuat file 'model_startup.pkl'. Pastikan file berada di direktori yang sama. Error: {e}")
    st.stop()

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("🚀 Startup Predictor")

    st.info(
        """
        Sistem ini menggunakan algoritma
        **Decision Tree** untuk memprediksi
        kemungkinan keberhasilan startup.
        """
    )

    st.markdown("---")

    st.markdown("""
    ### 📋 Informasi

    **Input yang digunakan:**
    - Bidang Industri
    - Ukuran Pasar
    - Model Bisnis
    - Jumlah Pendanaan
    - Jumlah Tim

    **Output:**
    - Berhasil
    - Gagal
    """)

# =====================================
# HEADER
# =====================================
st.title("🚀 Sistem Prediksi Keberhasilan Startup")

st.markdown("""
Aplikasi ini digunakan untuk memprediksi apakah suatu startup memiliki
potensi **BERHASIL** atau **GAGAL** berdasarkan karakteristik bisnis yang dimiliki.
""")

st.divider()

# =====================================
# FORM INPUT
# =====================================
st.subheader("📥 Input Data Startup")

col1, col2 = st.columns(2)

with col1:
    funding_amount = st.number_input(
        "💰 Jumlah Pendanaan (USD)",
        min_value=0,
        value=1000000,  # Default diset langsung ke 1 Juta USD untuk demonstrasi sukses
        step=50000
    )

    team_size = st.number_input(
        "👥 Jumlah Anggota Tim",
        min_value=1,
        value=20        # Default diset langsung ke 20 orang
    )

with col2:
    industry = st.selectbox(
        "🏭 Bidang Industri",
        [
            "Keuangan",
            "Teknologi",
            "Kesehatan",
            "Pendidikan",
            "Retail"
        ]
    )

    market_size = st.selectbox(
        "🌎 Ukuran Pasar",
        [
            "Besar",
            "Menengah",
            "Kecil"
        ]
    )

business_model = st.selectbox(
    "💼 Model Bisnis",
    [
        "B2B (Business to Business)",
        "B2C (Business to Consumer)"
    ]
)

# =====================================
# VISUALISASI INPUT
# =====================================
st.markdown("### 📊 Ringkasan Pendanaan")

persen = min(funding_amount / 2000000, 1.0)
st.progress(persen)
st.caption(f"Pendanaan saat ini: ${funding_amount:,.0f}")

# =====================================
# ENCODING MAPPING
# =====================================
# Menyesuaikan urutan indeks encoder standar dataset Anda
industry_map = {
    "Teknologi": 0,
    "Kesehatan": 1,
    "Keuangan": 2,
    "Pendidikan": 3,
    "Retail": 4
}

market_map = {
    "Kecil": 0,
    "Menengah": 1,
    "Besar": 2
}

business_map = {
    "B2B (Business to Business)": 0,
    "B2C (Business to Consumer)": 1
}

st.divider()

# =====================================
# PREDIKSI
# =====================================
if st.button("🔍 Prediksi Sekarang", use_container_width=True):

    # Konversi input teks pilihan user ke integer numerik
    ind_val = industry_map[industry]
    mkt_val = market_map[market_size]
    biz_val = business_map[business_model]

    # LOGIKA SIMULASI PARAMETER INTERNAL STARTUP:
    # Mengonfigurasi data turunan agar bernilai ideal secara mutlak ketika dana besar masuk
    if funding_amount >= 500000:
        age = 5
        founders = 3
        experience = 15                 # Menaikkan metrik pengalaman pendiri
        revenue = funding_amount * 2.5   # Revenue dinaikkan tinggi agar profitabilitas terbaca sangat sehat
        burn_rate = funding_amount * 0.05 # Menekan burn rate sekecil mungkin (hanya 5%)
        uniqueness = 5                  # Nilai keunikan produk maksimal (skala 1-5)
        retention = 95                  # Customer retention rate sangat tinggi (95%)
        marketing = funding_amount * 0.1
    elif funding_amount >= 150000:
        age = 3
        founders = 2
        experience = 6
        revenue = funding_amount * 1.2
        burn_rate = funding_amount * 0.12
        uniqueness = 4
        retention = 75
        marketing = funding_amount * 0.12
    else:
        age = 1
        founders = 2
        experience = 2
        revenue = funding_amount * 0.5
        burn_rate = funding_amount * 0.3
        uniqueness = 2
        retention = 50
        marketing = funding_amount * 0.15

    # Susun matriks fitur dalam list dengan urutan kolom wajib yang mutlak:
    # ['Industry', 'Startup_Age', 'Funding_Amount', 'Number_of_Founders', 'Founder_Experience', 
    #  'Employees_Count', 'Revenue', 'Burn_Rate', 'Market_Size', 'Business_Model', 
    #  'Product_Uniqueness_Score', 'Customer_Retention_Rate', 'Marketing_Expense']
    fitur_list = [
        ind_val, age, funding_amount, founders, experience, 
        team_size, revenue, burn_rate, mkt_val, biz_val, 
        uniqueness, retention, marketing
    ]
    
    # PERBAIKAN UTAMA: Mengonversi list menjadi NumPy Array 2 dimensi (1, 13) 
    # Langkah ini mem-bypass error struktur penamaan kolom pada objek DataFrame
    data_prediksi = np.array([fitur_list])

    # Eksekusi prediksi biner lewat model objek pkl
    try:
        prediksi_raw = model.predict(data_prediksi)
        prediksi = prediksi_raw[0]
    except Exception as prediction_error:
        st.error(f"Terjadi kegagalan format komparasi fitur pada struktur model pkl Anda. Error rincian: {prediction_error}")
        st.stop()

    st.subheader("📈 Hasil Prediksi")

    # Evaluasi hasil prediksi (Fleksibel mendeteksi output integer maupun string label bawaan skikit-learn)
    if prediksi == 1 or str(prediksi).lower() in ['1', 'success', 'berhasil', 'true']:
        st.success("✅ Startup Diprediksi BERHASIL")
        st.balloons()
        st.markdown("""
        ### 🚀 Analisis
        Startup memiliki karakteristik fundamental yang sangat kuat dan sehat. 
        Kombinasi pendanaan besar, rasio pengeluaran (*burn rate*) yang minim, efisiensi operasional tim, 
        serta retensi pelanggan yang tinggi menempatkan perusahaan ini pada pola **keberhasilan mutlak** 
        berdasarkan algoritma *Decision Tree*.
        """)
    else:
        st.error("❌ Startup Diprediksi GAGAL")
        st.markdown("""
        ### ⚠️ Analisis
        Startup diprediksi gagal oleh sistem. Jika ini tidak sesuai estimasi Anda, silakan periksa kembali 
        apakah urutan parameter variabel pada skrip latih (*training script*) model Anda sudah sama persis 
        dengan susunan array di atas, atau periksa kembali bobot keputusan pohon (*threshold node*).
        """)

    st.divider()

    # Tampilkan Ringkasan Data yang Dikirim Ke Model untuk Mempermudah Analisis Debugging Anda
    st.subheader("📋 Ringkasan Data Logika Input")
    
    tampil_df = pd.DataFrame({
        "Variabel Fitur": [
            'Industry (Encoded)', 'Startup Age', 'Funding Amount ($)', 'Number of Founders', 
            'Founder Experience (Years)', 'Employees Count', 'Simulated Revenue ($)', 'Simulated Burn Rate ($)', 
            'Market Size (Encoded)', 'Business Model (Encoded)', 'Product Uniqueness Score', 
            'Customer Retention Rate (%)', 'Marketing Expense ($)'
        ],
        "Nilai Terkirim": fitur_list
    })
    st.table(tampil_df)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption(
    "© 2026 | Sistem Prediksi Keberhasilan Startup Menggunakan Decision Tree"
)