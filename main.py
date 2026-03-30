import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset loading
data = pd.read_csv("dataset.csv")
X = data['symptoms']
y = data['disease']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Model train
model = MultinomialNB()
model.fit(X_vectorized, y)

# Advice Dictionary
advice_dict = {
    "Flu": "Take rest, drink warm fluids and take paracetamol if fever persists.",
    "Common Cold": "Drink warm water, take steam inhalation and rest properly.",
    "Migraine": "Rest in a quiet dark room and avoid bright lights.",
    "Food Poisoning": "Drink ORS and stay hydrated. Avoid oily food.",
    "Heart Disease": "Seek medical help immediately.",
    "Diabetes": "Monitor blood sugar regularly and avoid sugary foods.",
    "Dengue": "Drink plenty of fluids and consult doctor if symptoms worsen.",
    "Cancer": "Consult a specialist doctor immediately."
}

# UI
st.title("🩺 Disease Prediction System")
st.write("Enter your symptoms to predict possible disease.")
user_input = st.text_input("Enter Symptoms (example: fever cough headache)")

# Prediction button
if st.button("Predict Disease"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    predicted_disease = prediction[0]
    
    st.subheader("Predicted Disease")
    st.success(predicted_disease)

    advice = advice_dict.get(predicted_disease)
    st.subheader("Medical Advice")
    st.info(advice)