import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Risiko Penyakit Jantung")

# Input user
with st.form("heart_form"):
    age = st.number_input("Umur", 1, 120)
    sex = st.selectbox("Jenis Kelamin (0: Perempuan, 1: Laki-laki)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200)
    chol = st.number_input("Kolesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Ya, 0 = Tidak)", [0, 1])
    restecg = st.selectbox("Hasil EKG Istirahat (0–2)", [0, 1, 2])
    thalach = st.number_input("Denyut Jantung Maksimum", 60, 220)
    exang = st.selectbox("Angina Induced by Exercise (1 = Ya, 0 = Tidak)", [0, 1])
    oldpeak = st.number_input("Depresi ST", 0.0, 6.0)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Jumlah pembuluh darah besar (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (1: Normal, 2: Fixed Defect, 3: Reversible Defect)", [1, 2, 3])

    submitted = st.form_submit_button("Prediksi")

# Prediksi
if submitted:
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)[0]

    st.subheader("Hasil Prediksi:")
    if result == 1:
        st.error(" Berisiko Terkena Penyakit Jantung")
    else:
        st.success(" Tidak Berisiko Penyakit Jantung")
