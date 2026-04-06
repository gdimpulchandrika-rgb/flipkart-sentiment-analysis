import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("Flipkart Review Sentiment Analysis")

st.write("Enter a product review and the model will predict whether it is Positive or Negative.")

# User input
review = st.text_area("Enter your review:")

# Prediction
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        vector = tfidf.transform([review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("+ Positive Review")
        else:
            st.error(" - Negative Review")
