import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Heart Risk Predictor AI", page_icon="🫀", layout="centered"
)


# 2. Load Trained Model
@st.cache_resource
def load_model():
    model_data = joblib.load("heart_disease_model.joblib")
    return model_data["model"], model_data["features"]


try:
    model, feature_names = load_model()
except Exception as e:
    st.error(
        f"Error loading model file. Ensure 'heart_disease_model.joblib' is in the same folder. Details: {e}"
    )
    st.stop()

# 3. Header Section
st.title("🫀 Heart Disease Risk Prediction System")
st.markdown(
    """
This application uses an **Optuna-Tuned LightGBM Machine Learning Model** trained on clinical medical metrics to assess patient heart disease risk.
"""
)
st.write("---")

# 4. User Input Form
st.subheader("📋 Enter Patient Parameters")

input_data = {}

for feature in feature_names:
    feat_lower = feature.lower().strip()

    # --- SEX (Dropdown) ---
    if "sex" in feat_lower:
        sex_val = st.selectbox(
            f"Sex ({feature})",
            options=["Male", "Female"],
            help="Select patient biological sex",
        )
        input_data[feature] = 1 if sex_val == "Male" else 0

    # --- AGE (Integer) ---
    elif "age" in feat_lower:
        input_data[feature] = int(
            st.number_input(
                f"Age in Years ({feature})",
                min_value=1,
                max_value=120,
                value=50,
                step=1,
            )
        )

    # --- CHEST PAIN TYPE (Dropdown) ---
    elif "cp" in feat_lower or "chest" in feat_lower:
        cp_val = st.selectbox(
            f"Chest Pain Type ({feature})",
            options=[
                "0: Typical Angina",
                "1: Atypical Angina",
                "2: Non-anginal Pain",
                "3: Asymptomatic",
            ],
        )
        input_data[feature] = int(cp_val.split(":")[0])

    # --- FASTING BLOOD SUGAR (Dropdown) ---
    elif "fbs" in feat_lower or "sugar" in feat_lower:
        fbs_val = st.selectbox(
            f"Fasting Blood Sugar > 120 mg/dl ({feature})",
            options=["No", "Yes"],
        )
        input_data[feature] = 1 if fbs_val == "Yes" else 0

    # --- EXERCISE INDUCED ANGINA (Dropdown) ---
    elif "exang" in feat_lower or "angina" in feat_lower:
        exang_val = st.selectbox(
            f"Exercise Induced Angina ({feature})", options=["No", "Yes"]
        )
        input_data[feature] = 1 if exang_val == "Yes" else 0

    # --- RESTING ECG / EKG (Dropdown) ---
    elif (
        "restecg" in feat_lower
        or "ecg" in feat_lower
        or "ekg" in feat_lower
        or "resting electrocardiographic" in feat_lower
    ):
        restecg_val = st.selectbox(
            f"Resting ECG/EKG Results ({feature})",
            options=["0: Normal", "1: ST-T Wave Abnormality", "2: Hypertrophy"],
        )
        input_data[feature] = int(restecg_val.split(":")[0])

    # --- SLOPE (Dropdown) ---
    elif "slope" in feat_lower:
        slope_val = st.selectbox(
            f"Slope of Peak Exercise ST Segment ({feature})",
            options=["0: Upsloping", "1: Flat", "2: Downsloping"],
        )
        input_data[feature] = int(slope_val.split(":")[0])

    # --- THAL (Dropdown) ---
    elif "thal" in feat_lower and "thalach" not in feat_lower:
        thal_val = st.selectbox(
            f"Thalassemia ({feature})",
            options=[
                "0: Normal",
                "1: Fixed Defect",
                "2: Reversible Defect",
                "3: Other",
            ],
        )
        input_data[feature] = int(thal_val.split(":")[0])

    # --- NUMBER OF MAJOR VESSELS (Dropdown) ---
    elif "ca" in feat_lower or "vessel" in feat_lower:
        input_data[feature] = int(
            st.selectbox(
                f"Number of Major Vessels ({feature})", options=[0, 1, 2, 3, 4]
            )
        )

    # --- RESTING BLOOD PRESSURE (Integer) ---
    elif "trestbps" in feat_lower or "bp" in feat_lower or "pressure" in feat_lower:
        input_data[feature] = int(
            st.number_input(
                f"Resting Blood Pressure in mm Hg ({feature})",
                min_value=50,
                max_value=250,
                value=120,
                step=1,
            )
        )

    # --- CHOLESTEROL (Integer) ---
    elif "chol" in feat_lower:
        input_data[feature] = int(
            st.number_input(
                f"Serum Cholesterol in mg/dl ({feature})",
                min_value=50,
                max_value=600,
                value=200,
                step=1,
            )
        )

    # --- MAXIMUM HEART RATE / THALACH (Integer) ---
    elif (
        "thalach" in feat_lower
        or "maxhr" in feat_lower
        or "heart rate" in feat_lower
        or "hr" in feat_lower
    ):
        input_data[feature] = int(
            st.number_input(
                f"Maximum Heart Rate Achieved in BPM ({feature})",
                min_value=50,
                max_value=230,
                value=150,
                step=1,
            )
        )

    # --- ST DEPRESSION / OLDPEAK (Float Decimal) ---
    elif "oldpeak" in feat_lower or "st depression" in feat_lower:
        input_data[feature] = float(
            st.number_input(
                f"ST Depression - Oldpeak ({feature})",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
        )

    # --- FALLBACK ---
    else:
        input_data[feature] = int(
            st.number_input(f"Value for {feature}", value=0, step=1)
        )

st.write("---")

# 5. Prediction Execution
if st.button("🔍 Analyze Heart Risk"):
  input_df = pd.DataFrame([input_data])[feature_names]
  prediction_prob = model.predict_proba(input_df)[0][1]

  st.subheader("📊 Diagnostic Result")
  st.write(f"**Heart Disease Probability:** `{prediction_prob * 100:.2f}%`")
  st.progress(float(prediction_prob))

  # Updated 3-Tier Risk Logic
  if prediction_prob >= 0.65:
    st.error(
        "🔴 **HIGH RISK:** The model indicates a high likelihood of Heart"
        " Disease presence. Further clinical evaluation is recommended."
    )
  elif 0.35 <= prediction_prob < 0.65:
    st.warning(
        "🟡 **MODERATE RISK:** The model indicates borderline risk factors."
        " Patient monitoring and lifestyle recommendations are suggested."
    )
  else:
    st.success(
        "🟢 **LOW RISK:** The model indicates a lower likelihood of Heart"
        " Disease presence."
    )