
import streamlit as st
from components.text_cleaning import clean_text
from components.vectorizer import semantic_vectorize
from components.similarity import cosine_sim
from components.data_loader import load_data
from components.ranker import rank_resumes

st.set_page_config(page_title="Smart Resume Matcher", layout="wide")

st.title("📄 Smart Resume Matcher")
st.write("Check how well a resume matches a job description using NLP")

# -----------------------------
# INPUT SECTION
# -----------------------------
job_desc = st.text_area("📝 Paste Job Description", height=200)
resume_text = st.text_area("📄 Paste Resume", height=200)

# -----------------------------
# SINGLE RESUME MATCHING
# -----------------------------
if st.button("Check Resume Match"):
    if job_desc.strip() == "" or resume_text.strip() == "":
        st.warning("Please paste both Job Description and Resume")
    else:
        clean_job = clean_text(job_desc)
        clean_resume = clean_text(resume_text)

        embeddings = semantic_vectorize([clean_resume, clean_job])

        score = cosine_sim(embeddings[0], embeddings[1]) * 100

        st.success("✅ Matching Completed")
        st.metric("Resume Match Score", f"{score:.2f} %")

        if score >= 75:
            st.write("🟢 **Strong Match**")
        elif score >= 50:
            st.write("🟡 **Moderate Match**")
        else:
            st.write("🔴 **Low Match**")

# 