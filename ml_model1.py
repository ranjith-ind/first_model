import streamlit as st
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Exam Score Predictor", layout="wide")

# Title and description

st.title("📚 Exam Score Predictor")
st.markdown("Predict exam scores based on study hours and sleep hours using Linear Regression")

# Load and train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("study_vs_marks.csv")
    x = df[["study_hours", "sleep_hours"]]
    y = df["exam_score"]
    model = LinearRegression()
    model.fit(x, y)
    return model, df

model, df = train_model()

# Display model information
with st.expander("📊 Model Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Study Hours Coefficient", f"{model.coef_[0]:.4f}")
    with col2:
        st.metric("Sleep Hours Coefficient", f"{model.coef_[1]:.4f}")
    st.info(f"**Intercept:** {model.intercept_:.4f}")

# Create two columns for sliders
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Study Hours",
        min_value=float(df["study_hours"].min()),
        max_value=float(df["study_hours"].max()),
        value=5.0,
        step=0.5,
        help="Select the number of study hours"
    )

with col2:
    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=float(df["sleep_hours"].min()),
        max_value=float(df["sleep_hours"].max()),
        value=7.0,
        step=0.5,
        help="Select the number of sleep hours"
    )

# Make prediction
prediction = model.predict([[study_hours, sleep_hours]])[0]

# Display prediction
st.success(f"## Predicted Exam Score: {prediction:.2f}")

# Display data statistics
with st.expander("📈 Dataset Statistics"):
    st.write(df.describe())

# Display data visualization
with st.expander("📉 Data Visualization"):
    st.scatter_chart(data=df.set_index("study_hours")[["exam_score"]])

# Display sample predictions
with st.expander("🔮 Sample Predictions"):
    sample_data = df.head(5).copy()
    sample_predictions = model.predict(sample_data[["study_hours", "sleep_hours"]])
    sample_data["Predicted Score"] = sample_predictions
    st.dataframe(sample_data, use_container_width=True)

