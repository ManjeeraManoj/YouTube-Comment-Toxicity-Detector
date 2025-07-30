# toxicity_detector_full.py

import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# -------- Data Cleaning Function -------- #
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    return text

# -------- Model Training -------- #
try:
    model = pickle.load(open("toxicity_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except:
    st.info("Training model for the first time...")
    df = pd.read_csv("train.csv")[['comment_text', 'toxic']]
    df['comment_text'] = df['comment_text'].apply(clean_text)

    X = df['comment_text']
    y = df['toxic']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    pickle.dump(model, open("toxicity_model.pkl", "wb"))
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

# -------- Streamlit App -------- #
st.title("\U0001F9E0 YouTube Comment Toxicity Detector")

user_input = st.text_area("Enter a YouTube comment to check for toxicity:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned])
    result = model.predict(vec_input)

    if result[0] == 1:
        st.error("\u274C Toxic Comment Detected")
    else:
        st.success("\u2705 Comment is Non-Toxic")
