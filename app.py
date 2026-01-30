import streamlit as st
import pickle
import re
import os
import pdfplumber
from docx import Document
import numpy as np
from scipy.sparse import hstack, csr_matrix
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
# Load Model & Vectorizer
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# =========================
# Text Cleaning
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# =========================
# Skill Overlap
# =========================
def skill_overlap(resume, job):
    r, j = set(resume.split()), set(job.split())
    return len(r & j) / (len(j) + 1)

# =========================
# Resume Text Extraction
# =========================
def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text

def extract_docx(file):
    doc = Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def extract_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def load_resume(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_docx(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return extract_txt(uploaded_file)
    else:
        return ""

# =========================
# Prediction
# =========================
def predict_match(resume_text, job_text):
    r = clean_text(resume_text)
    j = clean_text(job_text)

    vec = vectorizer.transform([r + " " + j])
    overlap = csr_matrix([[skill_overlap(r, j)]])
    X = hstack([vec, overlap])

    score = model.predict(X)[0]
    return round(score * 100, 2)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ATS Resume Matcher", layout="centered")

st.title("📄 ATS Resume & Job Matching System")
st.write("Upload your resume and paste the job description to get an ATS score.")

uploaded_file = st.file_uploader(
    "Upload Resume (.pdf, .docx, .txt)",
    type=["pdf", "docx", "txt"]
)

job_description = st.text_area(
    "Paste Job Description Here",
    height=200
)

if st.button("🔍 Calculate ATS Score"):
    if uploaded_file and job_description.strip():
        with st.spinner("Analyzing resume..."):
            resume_text = load_resume(uploaded_file)
            score = predict_match(resume_text, job_description)

        st.success(f"✅ ATS Match Score: **{score}%**")

        if score >= 75:
            st.info("🎯 Strong match – Highly suitable for this role.")
        elif score >= 50:
            st.warning("⚠️ Moderate match – Consider improving skills.")
        else:
            st.error("❌ Low match – Resume needs improvement.")

    else:
        st.error("Please upload a resume and enter job description.")
