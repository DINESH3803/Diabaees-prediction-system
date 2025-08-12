import pandas as pd
import joblib

# Load the saved scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('diabetes_model.pkl')

# New patient data
new_patient = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 30,
    'Insulin': 100,
    'BMI': 28.0,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 45
}

# Create a DataFrame
new_df = pd.DataFrame([new_patient])

# Apply the same scaling transformation
new_df_scaled = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)

# Predict
prediction = model.predict(new_df_scaled)[0]

print(f"Predicted Outcome: {prediction} ({'Diabetic' if prediction == 1 else 'Not Diabetic'})")