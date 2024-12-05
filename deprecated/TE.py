import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from collections import defaultdict
import math
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Read your CSV file with synopses and genres
df = pd.read_csv('english_movies_clean.csv')  # Replace with your file path
df['genres'] = df['genres'].apply(lambda x: x.split(', '))  # Assuming genres are comma-separated


df['cleaned_synopsis'] = df['overview']

# 3. Convert genres to a binary format (multi-label) for classification
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])  # Multi-hot encoding of genres

# 4. Set the number of documents per genre to sample
max_docs_per_genre = 1  # Specify the number of documents you want to sample per genre

# 5. Sample a fixed number of documents per genre
sampled_df = pd.DataFrame()
for genre in mlb.classes_:
    genre_df = df[df['genres'].apply(lambda genres: genre in genres)]
    sampled_df = pd.concat([sampled_df, genre_df.sample(n=min(max_docs_per_genre, len(genre_df)), random_state=42)])

y = mlb.fit_transform(sampled_df['genres'])
# 6. TF-IDF calculation for each genre
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(sampled_df['cleaned_synopsis'])
print(X_tfidf)

# 7. Chi-Square calculation for each genre
def chi_square(X, y):
    chi2_scores = {}
    for genre_idx in range(y.shape[1]):  # For each genre
        genre_data = y[:, genre_idx]  # Binary vector for that genre
        chi2_values, p_values = chi2(X, genre_data)
        chi2_scores[mlb.classes_[genre_idx]] = chi2_values
    return chi2_scores

# Ensure the data types and dimensions of X and y match
print("Shape of X_tfidf:", X_tfidf.shape)  # (n_samples, n_features)
print("Shape of y:", y.shape)  # (n_samples, n_genres)

# Compute Chi-Square scores
chi2_scores = chi_square(X_tfidf, y)

# 8. C-Value Calculation for each term (optional, but often useful)
def compute_c_value(terms, term_frequencies, doc_count):
    c_values = {}
    for term in terms:
        freq_in_docs = term_frequencies.get(term, 0)
        c_value = freq_in_docs * math.log(doc_count / (1 + freq_in_docs))
        c_values[term] = c_value
    return c_values

# Calculate term frequencies for C-Value computation
term_frequencies = defaultdict(int)
for doc in sampled_df['cleaned_synopsis']:
    for term in doc.split():
        term_frequencies[term] += 1

# Get total document count
doc_count = len(sampled_df)

# Get C-Value for each term
c_values = compute_c_value(tfidf_vectorizer.get_feature_names_out(), term_frequencies, doc_count)

# 9. Combine TF-IDF, Chi-Square, and C-Value for each genre
combined_scores = defaultdict(lambda: defaultdict(float))  # {genre: {term: score}}



# Combine scores for each term and each genre
for genre_idx in range(y.shape[1]):
    genre = mlb.classes_[genre_idx]

    # Get term scores for the genre
    genre_tfidf_scores = X_tfidf[:, genre_idx].toarray().flatten()


    genre_chi2_scores = chi2_scores[genre]
    
    for idx, term in enumerate(tfidf_vectorizer.get_feature_names_out()):
        print(idx, term)
        print(genre_tfidf_scores)
        # Combine scores: TF-IDF + Chi-Square + C-Value (weighted sum)
        combined_score = (0.5 * genre_tfidf_scores[idx] + 0.3 * genre_chi2_scores[idx] + 0.2 * c_values.get(term, 0))
        combined_scores[genre][term] = combined_score

# 10. Rank terms by combined score for each genre
ranked_terms_by_genre = {}
for genre, terms in combined_scores.items():
    # Sort terms for this genre by the combined score
    ranked_terms_by_genre[genre] = sorted(terms.items(), key=lambda x: x[1], reverse=True)

# 11. Output top N terms for each genre
top_n = 10
for genre, ranked_terms in ranked_terms_by_genre.items():
    print(f"\nTop {top_n} terms for {genre} genre:")
    for term, score in ranked_terms[:top_n]:
        print(f"{term}: {score}")
