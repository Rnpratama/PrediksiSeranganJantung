import streamlit as st
import numpy as np
import joblib

# === Load model dan scaler ===
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# === Judul Aplikasi ===
st.set_page_config(page_title="Prediksi Serangan Jantung")
st.title("Prediksi Risiko Penyakit Jantung")
st.markdown("Masukkan data pasien untuk memprediksi kemungkinan risiko penyakit jantung.")

# === Form Input User ===
with st.form("form_heart"):
    age = st.number_input("Umur", min_value=1, max_value=120, value=50)
    sex = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    sex_val = 1 if sex == "Laki-laki" else 0

    cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", 80, 200, value=120)
    chol = st.number_input("Kolesterol (mg/dl)", 100, 600, value=200)
    fbs = st.radio("Gula Darah Puasa > 120 mg/dl?", ["Ya", "Tidak"])
    fbs_val = 1 if fbs == "Ya" else 0

    restecg = st.selectbox("Hasil EKG Istirahat (restecg)", [0, 1, 2])
    thalach = st.number_input("Detak Jantung Maksimum (thalach)", 60, 220, value=150)
    exang = st.radio("Angina Karena Olahraga (exang)", ["Ya", "Tidak"])
    exang_val = 1 if exang == "Ya" else 0

    oldpeak = st.number_input("Penurunan ST saat olahraga (oldpeak)", 0.0, 6.0, step=0.1, value=1.0)
    slope = st.selectbox("Kemiringan ST (slope)", [0, 1, 2])
    ca = st.selectbox("Jumlah pembuluh darah utama (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi Sekarang")

# === Prediksi ===
if submit:
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                            restecg, thalach, exang_val, oldpeak, slope,
                            ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("Pasien **berisiko** terkena penyakit jantung.")
    else:
        st.success("Pasien **tidak berisiko** terkena penyakit jantung.")
