# Diabetes Prediction System

This project is an advanced diabetes prediction system using machine learning models. It provides both a Streamlit web app for interactive predictions and Python scripts for batch and single patient predictions.

## Project Structure

- `app.py` – Streamlit web application for single/batch predictions and model analysis.
- `regression_model.py` – Model training script (Logistic Regression & Random Forest), data preprocessing, and model selection.
- `test.py` – Script for predicting diabetes for a single patient using saved models.
- `diabetes.csv` – Dataset used for training.
- `sample.csv` – Example CSV for batch prediction.
- `diabetes_model.pkl`, `logistic_regression_model.pkl`, `random_forest_model.pkl` – Saved models.
- `scaler.pkl`, `standard_scaler.pkl` – Saved scalers for feature normalization.

## Setup

1. **Install dependencies:**
   ```sh
   pip install pandas scikit-learn streamlit joblib matplotlib shap
   ```

2. **Train models (if not already trained):**
   ```sh
   python regression_model.py
   ```

3. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

## Usage

### Web App (`app.py`)
- **Single Patient Prediction:** Enter patient details to predict diabetes risk.
- **Batch Prediction:** Upload a CSV file with patient data for batch predictions.
- **Model Analysis:** View feature importance for trained models.

### Script (`test.py`)
- Predict diabetes for a single patient by editing the patient data in the script and running:
  ```sh
  python test.py
  ```

## Data Format

**CSV columns required:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Notes

- Models and scalers are saved as `.pkl` files after training.
- The app uses the best-performing model (Random Forest or Logistic Regression) based on F1 and AUC scores.
- For educational purposes only. Not medical advice.

## License

MIT License
