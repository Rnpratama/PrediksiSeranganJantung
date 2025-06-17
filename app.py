import streamlit as st
import numpy as np
import joblib
import os

st.title("Prediksi Risiko Penyakit Jantung")

# Load model dan scaler dengan error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("heart_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        return None, None

model, scaler = load_model()

# Input user
with st.form("heart_form"):
    age = st.number_input("Umur", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Jenis Kelamin (0: Perempuan, 1: Laki-laki)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", min_value=80, max_value=200, value=120)
    chol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Ya, 0 = Tidak)", [0, 1])
    restecg = st.selectbox("Hasil EKG Istirahat (0–2)", [0, 1, 2])
    thalach = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Angina Induced by Exercise (1 = Ya, 0 = Tidak)", [0, 1])
    oldpeak = st.number_input("Depresi ST", min_value=0.0, max_value=6.0, value=1.0, step=0.1, format="%.1f")
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Jumlah pembuluh darah besar (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (1: Normal, 2: Fixed Defect, 3: Reversible Defect)", [1, 2, 3])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    if (model is None) or (scaler is None):
        st.error("Model/scaler belum dimuat, prediksi tidak dapat dilakukan.")
    else:
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        scaled_features = scaler.transform(features)
        result = model.predict(scaled_features)[0]

        if result == 1:
            st.error("Berisiko Terkena Penyakit Jantung")
        else:
            st.success("Tidak Berisiko Penyakit Jantung")
