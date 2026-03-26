from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import streamlit as st
df=pd.read_csv(r"study_time_vs_marks_dataset.csv")
x=df[["study_hours","sleep_hours"]]
y=df["exam_score"]
model=LinearRegression()
model.fit(x,y)  
user_input1=int(input("Enter study hours:" ))
user_input2=int(input("Enter sleep hours:" ))
input_df = pd.DataFrame([[user_input1, user_input2]], columns=["study_hours", "sleep_hours"])
predicted_score = model.predict(input_df)
print(f"Predicted exam score: {predicted_score[0]:.2f}")

# Streamlit Interface
st.set_page_config(page_title="Exam Score Predictor", page_icon="📘", layout="centered")
st.title("📘 Exam Score Predictor")
st.write("Predict exam score using study hours and sleep hours.")

st_sub_col1, st_sub_col2 = st.columns(2)
with st_sub_col1:
	study_hours = st.slider("Study hours", min_value=0, max_value=14, value=5, step=1)
with st_sub_col2:
	sleep_hours = st.slider("Sleep hours", min_value=0, max_value=12, value=7, step=1)

display_mode = st.selectbox("Output format", ["Score only", "Score with performance level"])

if st.button("Predict score"):
	st_input_df = pd.DataFrame([[study_hours, sleep_hours]], columns=["study_hours", "sleep_hours"])
	st_predicted_score = model.predict(st_input_df)[0]
	st.success(f"Predicted exam score: {st_predicted_score:.2f}")

	if display_mode == "Score with performance level":
		if st_predicted_score >= 85:
			level = "Excellent"
		elif st_predicted_score >= 70:
			level = "Good"
		elif st_predicted_score >= 50:
			level = "Average"
		else:
			level = "Needs Improvement"
		st.info(f"Performance level: {level}")
