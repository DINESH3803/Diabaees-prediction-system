import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
import numpy as np
import shap

# -------------------------------
# Load Models and Scaler
# -------------------------------
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load("logistic_regression_model.pkl")
        rf_model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return logistic_model, rf_model, scaler
    except FileNotFoundError:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return None, model, scaler

logistic_model, rf_model, scaler = load_models()
model_to_use = rf_model if rf_model is not None else logistic_model

# Initialize SHAP explainer for the loaded model
explainer = shap.TreeExplainer(model_to_use)

# Feature names consistent with training data
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# -------------------------------
# Streamlit Config & Header
# -------------------------------
st.set_page_config(page_title="Advanced Diabetes Prediction App", page_icon="🩺", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🩺 Advanced Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Patient", "📊 Batch Prediction", "📈 Model Analysis"])

# ================================
# TAB 1 – SINGLE PATIENT PREDICTION
# ================================
with tab1:
    st.subheader("Enter Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0.0, 250.0, 100.0)
        blood_pressure = st.number_input("Blood Pressure", 0.0, 150.0, 70.0)
        skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
    with col2:
        insulin = st.number_input("Insulin Level", 0.0, 900.0, 80.0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
        age = st.number_input("Age", 0, 120, 30)

    if st.button("🔍 Predict Diabetes Risk"):
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        # Scale input and convert to DataFrame with feature names
        scaled_array = scaler.transform(input_data)
        scaled_df = pd.DataFrame(scaled_array, columns=feature_names)

        # Predict using the selected model
        prediction = model_to_use.predict(scaled_df)[0]
        probability = model_to_use.predict_proba(scaled_df)[0]

        # Display prediction result
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.markdown(
                    "<div style='padding:20px; background-color:#ffebee; border-left:5px solid #f44336; border-radius:5px;'>"
                    "<h3 style='color:#d32f2f; margin:0;'>⚠️ High Risk</h3>"
                    "<p style='margin:5px 0 0 0;'>Patient is likely diabetic</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='padding:20px; background-color:#e8f5e8; border-left:5px solid #4caf50; border-radius:5px;'>"
                    "<h3 style='color:#388e3c; margin:0;'>✅ Low Risk</h3>"
                    "<p style='margin:5px 0 0 0;'>Patient is unlikely diabetic</p></div>",
                    unsafe_allow_html=True
                )
        with col2:
            st.metric("Diabetes Probability", f"{probability[1]:.2%}")
            st.metric("Non-Diabetes Probability", f"{probability[0]:.2%}")

# ================================
# TAB 2 – BATCH CSV PREDICTION
# ================================
with tab2:
    st.subheader("Upload CSV File for Batch Predictions")
    with st.expander("📋 Expected CSV Format"):
        sample_data = pd.DataFrame({
            "Pregnancies": [6, 1, 8],
            "Glucose": [148, 85, 183],
            "BloodPressure": [72, 66, 64],
            "SkinThickness": [35, 29, 0],
            "Insulin": [0, 0, 0],
            "BMI": [33.6, 26.6, 23.3],
            "DiabetesPedigreeFunction": [0.627, 0.351, 0.672],
            "Age": [50, 31, 32]
        })
        st.dataframe(sample_data)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ File loaded: {len(df)} patients")
                scaled = scaler.transform(df[feature_names])
                scaled_df_batch = pd.DataFrame(scaled, columns=feature_names)
                preds = model_to_use.predict(scaled_df_batch)
                probs = model_to_use.predict_proba(scaled_df_batch)

                df["Prediction"] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
                df["Risk_Probability"] = [f"{p[1]:.2%}" for p in probs]

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Patients", len(df))
                c2.metric("High Risk", sum(preds))
                c3.metric("Low Risk", len(df) - sum(preds))

                st.markdown("### 📋 Predictions Preview")
                st.dataframe(df, use_container_width=True)

                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    buf.getvalue(),
                    "diabetes_predictions.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# ================================
# TAB 3 – MODEL ANALYSIS
# ================================
with tab3:
    st.subheader("Model Feature Importance Analysis")
    options = []
    if logistic_model: options.append("Logistic Regression")
    if rf_model: options.append("Random Forest")

    if options:
        choice = st.selectbox("Choose Model:", options)

        def plot_importance(vals, feats, name):
            df_imp = pd.DataFrame({"Feature": feats, "Importance": vals}).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="#4caf50")
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f"{bar.get_width():.3f}", va="center")
            ax.set_title(f"{name} Feature Importance")
            return fig, df_imp.sort_values("Importance", ascending=False)

        if choice == "Random Forest":
            vals = rf_model.feature_importances_
            fig, df_imp = plot_importance(vals, feature_names, "Random Forest")
        else:
            vals = np.abs(logistic_model.coef_[0])
            fig, df_imp = plot_importance(vals, feature_names, "Logistic Regression")

        st.pyplot(fig)
        st.markdown("### 🎯 Top Features")
        st.dataframe(df_imp.head(5), use_container_width=True)
    else:
        st.error("No models available for analysis.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "<strong>Disclaimer:</strong> This tool is for educational purposes only. Not medical advice."
    "</div>",
    unsafe_allow_html=True
)
