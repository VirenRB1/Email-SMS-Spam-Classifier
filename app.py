import re
import streamlit as st
import pickle


# This must exist before unpickling transform.pkl,
# because the pickled object expects a `transform_text` name
# to be available in the module it is loaded in.
def transform_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# Load the model and vectorizer
transform_text = pickle.load(open("transform.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_text = st.text_input("Enter the text to classify")


# Create a function to predict the label of the input text
def predict(text):
    # Transform the input text
    transformed_text = transform_text(text)
    # Vectorize the transformed text
    vectorized_text = tfidf.transform([transformed_text])
    # Predict the label
    prediction = model.predict(vectorized_text)[0]
    return prediction


if st.button("Predict"):
    result = predict(input_text)
    if result == 1:
        st.header("Spam.")
    else:
        st.header("Not Spam.")
