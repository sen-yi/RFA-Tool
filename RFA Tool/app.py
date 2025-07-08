import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")  # Hide SHAP warnings

# Load your trained model
model = joblib.load("rf_model.pkl")

st.title("RFA Recurrence Risk Predictor")

# Input sliders
tumor_size = st.slider("Tumor Size (cm)", 0.5, 10.0, 3.0)
wattage = st.slider("Wattage", 10, 100, 40)
duration = st.slider("Duration (s)", 10, 300, 90)
margin = st.slider("Ablation Margin (mm)", 0.0, 10.0, 2.0)
deviation = st.slider("Needle Deviation (mm)", 0.0, 10.0, 1.0)

# Input as DataFrame
input_data = pd.DataFrame([[tumor_size, wattage, duration, margin, deviation]],
    columns=["Tumor_Size", "Wattage", "Duration", "Margin", "Deviation"])

# Predict
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display prediction
st.subheader(f"Predicted Recurrence: {'Yes' if prediction == 1 else 'No'}")
st.write(f"Estimated Probability: {probability:.2%}")
st.warning("High recurrence risk.") if probability > 0.5 else st.success("Low recurrence risk.")

# --- SHAP Explanation (silent + clean) ---
try:
    background = pd.DataFrame([input_data.mean()], columns=input_data.columns)
    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_data)

    # Pick class 1 if it's multiclass
    if shap_values.values.ndim == 3:
        impacts = shap_values[:, :, 1].values[0]
    else:
        impacts = shap_values.values[0]

    impacts = np.array(impacts).flatten()
    features = input_data.columns.tolist()

    # Make DataFrame
    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact": impacts
    })
    shap_df["Impact_abs"] = np.abs(shap_df["Impact"])
    shap_df = shap_df.sort_values("Impact_abs", ascending=False)

    # Display top 3
    st.subheader("Why this prediction?")
    for _, row in shap_df.head(3).iterrows():
        arrow = "⬆️" if row["Impact"] > 0 else "⬇️"
        direction = "increased" if row["Impact"] > 0 else "decreased"
        st.write(f"• **{row['Feature']}** {arrow} {direction} the recurrence risk")

except Exception as e:
    st.error("Could not generate SHAP explanation.")
    st.text(str(e))
