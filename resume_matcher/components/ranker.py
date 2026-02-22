import pandas as pd
from .similarity import cosine_sim

def rank_resumes(resume_vectors, job_vector, resumes_df):
    scores = []

    for i, vec in enumerate(resume_vectors):
        score = cosine_sim(vec, job_vector)
        scores.append(score)

    resumes_df["match_score"] = scores
    resumes_df["match_percentage"] = resumes_df["match_score"] * 100

    return resumes_df.sort_values(by="match_percentage", ascending=False)
