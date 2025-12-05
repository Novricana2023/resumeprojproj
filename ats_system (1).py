
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# 1. Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^ɗ\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces

    words = text.split()  # Tokenize
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords

    return ' '.join(words)

# 2. BERT Model and Embedding Function
# Load the pre-trained BERT model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

# 3. Keyword Extraction Setup (Vectorizers)
# Instantiate TfidfVectorizer and CountVectorizer globally for reusability
tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()

# Placeholder for fitting vectorizers - these would ideally be fitted on a large corpus
# and then saved/loaded, but for this demo, we'll refit them on combined sample data
# or ensure they are fit when the script is run in a standalone context

# 4. Keyword Extraction Functions
def extract_keywords_tfidf(tfidf_vectorizer_obj, tfidf_matrix, top_n=5):
    feature_names = tfidf_vectorizer_obj.get_feature_names_out()
    keywords_list = []
    for i in range(tfidf_matrix.shape[0]):  # Iterate through each document
        row = tfidf_matrix[i, :].toarray().flatten()
        top_n_indices = row.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_n_indices if row[idx] > 0]
        keywords_list.append(top_keywords)
    return keywords_list

def extract_keywords_countvectorizer(count_vectorizer_obj, count_matrix, top_n=5):
    feature_names = count_vectorizer_obj.get_feature_names_out()
    keywords_list = []
    for i in range(count_matrix.shape[0]):  # Iterate through each document
        row = count_matrix[i, :].toarray().flatten()
        top_n_indices = row.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_n_indices if row[idx] > 0]
        keywords_list.append(top_keywords)
    return keywords_list

# 5. Similarity Scoring Function
def calculate_cosine_similarity(embeddings1, embeddings2):
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    return similarity_matrix

# 6. Missing Keywords Identification
def identify_missing_keywords(resume_keywords_list, job_desc_keywords_list):
    resume_set = set(resume_keywords_list)
    job_desc_set = set(job_desc_keywords_list)
    missing = list(job_desc_set - resume_set)
    return missing

# 7. Feedback Generation
def generate_feedback(similarity_score, missing_keywords):
    feedback = f"Overall Resume-Job Description Match Score: {similarity_score:.2f} (out of 1.00)\n\n"

    if similarity_score >= 0.7:
        feedback += "Excellent match! Your resume aligns very well with the job description. " \
                    "Consider fine-tuning minor details for an even stronger application.\n"
    elif similarity_score >= 0.5:
        feedback += "Good match. Your resume shows strong relevance to the job. " \
                    "Focus on strengthening specific areas for better alignment.\n"
    else:
        feedback += "Moderate match. There's potential for improvement to better align " \
                    "your resume with the job description. Consider enhancing key sections.\n"

    if missing_keywords:
        feedback += "\nTo improve your match, consider incorporating the following keywords, " \
                    "which were prominent in the job description but not explicitly found in your resume: "
        feedback += ", ".join(missing_keywords) + ".\n"
    else:
        feedback += "\nNo significant missing keywords identified from the job description. " \
                    "Your resume covers key terms effectively.\n"

    feedback += "\nRemember to tailor your resume's experience and skills sections to highlight these areas."
    return feedback

# 8. Main ATS Analysis Report Function
def get_ats_analysis_report(resume_text_raw, job_desc_text_raw, tfidf_vectorizer_fit, count_vectorizer_fit):
    # a. Preprocess texts
    cleaned_resume = preprocess_text(resume_text_raw)
    cleaned_job_desc = preprocess_text(job_desc_text_raw)

    # b. Generate semantic embeddings
    resume_embedding = get_embeddings([cleaned_resume])
    job_desc_embedding = get_embeddings([cleaned_job_desc])

    # c. Calculate cosine similarity
    similarity_score = calculate_cosine_similarity(resume_embedding, job_desc_embedding)[0][0]

    # d. Transform cleaned texts into TF-IDF matrices using the already fitted vectorizer
    resume_tfidf_single = tfidf_vectorizer_fit.transform([cleaned_resume])
    job_desc_tfidf_single = tfidf_vectorizer_fit.transform([cleaned_job_desc])

    # e. Extract keywords from single-document TF-IDF matrices
    resume_keywords_single = extract_keywords_tfidf(tfidf_vectorizer_fit, resume_tfidf_single)[0]
    job_desc_keywords_single = extract_keywords_tfidf(tfidf_vectorizer_fit, job_desc_tfidf_single)[0]

    # f. Identify missing keywords
    missing_keywords = identify_missing_keywords(resume_keywords_single, job_desc_keywords_single)

    # g. Generate constructive feedback
    feedback = generate_feedback(similarity_score, missing_keywords)

    # h. Return a dictionary containing all relevant information
    report = {
        "original_resume_text": resume_text_raw,
        "original_job_description_text": job_desc_text_raw,
        "cleaned_resume_text": cleaned_resume,
        "cleaned_job_description_text": cleaned_job_desc,
        "resume_keywords_tfidf": resume_keywords_single,
        "job_description_keywords_tfidf": job_desc_keywords_single,
        "semantic_similarity_score": float(similarity_score),
        "missing_keywords": missing_keywords,
        "feedback": feedback
    }
    return report

print("ats_system.py created successfully.")
