from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_vectorize(texts):
    return model.encode(texts)
