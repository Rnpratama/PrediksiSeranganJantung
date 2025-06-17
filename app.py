import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔎 Prediksi Risiko Penyakit Jantung")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi apakah berisiko terkena penyakit jantung atau tidak.")

# Form input
with st.form("heart_form"):
    age = st.number_input("Umur", 1, 120)
    sex = st.selectbox("Jenis Kelamin", [("Perempuan", 0), ("Laki-laki", 1)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Tipe Nyeri Dada (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", 80, 200)
    chol = st.number_input("Kolesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", [("Tidak", 0), ("Ya", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Hasil EKG Istirahat (0–2)", [0, 1, 2])
    thalach = st.number_input("Denyut Jantung Maksimum", 60, 220)
    exang = st.selectbox("Angina Induced by Exercise?", [("Tidak", 0), ("Ya", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("Depresi ST akibat olahraga", 0.0, 6.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Besar Terlihat (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1: Normal, 2: Fixed, 3: Reversible)", [1, 2, 3])

    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi saat tombol ditekan
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error(" Pasien **berisiko** terkena penyakit jantung.")
    else:
        st.success("Pasien **tidak berisiko** terkena penyakit jantung.")
