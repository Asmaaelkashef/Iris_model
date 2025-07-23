import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")

st.title("🌼 Iris Flower Prediction App")

iris = load_iris()
X = iris.data
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

st.subheader("🌟 Prediction")
st.write(f"النوع المتوقع: **{iris.target_names[prediction[0]].capitalize()}**")

st.subheader("📊 Probability")
df_proba = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(df_proba.T)

st.markdown("---")
st.caption("By Asmaa Elkashef 💻 | Using scikit-learn + Streamlit")
