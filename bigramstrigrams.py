import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv

# Load the movie corpus
corpus_file = "english_movies.csv"  # Replace with your CSV file path
df = pd.read_csv(corpus_file)

# Ensure the 'overview' column exists and drop any rows where it's missing
if 'overview' not in df.columns:
    raise ValueError("The CSV does not have an 'overview' column.")
df = df.dropna(subset=['overview'])

# Extract the overview column
overviews = df['overview'].tolist()


def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    :param text: A string (sentence or document).
    :param n: The n in n-gram (e.g., 2 for bigram, 3 for trigram).
    :return: A list of n-grams as tuples.
    """
    tokens = text.split()  # Split the text into words
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# Generate bigrams and trigrams for each overview and keep results per entry
entries_bigrams_trigrams = []

for i, overview in enumerate(overviews):
    bigrams = generate_ngrams(overview, 2)
    trigrams = generate_ngrams(overview, 3)
    entries_bigrams_trigrams.append({
        "index": i,
        "overview": overview,
        "bigrams": bigrams,
        "trigrams": trigrams
    })

# Save the bigrams and trigrams for each entry to a CSV file
with open('entry_bigrams_trigrams.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Overview', 'Bigrams', 'Trigrams'])
    for entry in entries_bigrams_trigrams:
        writer.writerow([
            entry['index'],
            entry['overview'],
            ", ".join([" ".join(bigram) for bigram in entry['bigrams']]),
            ", ".join([" ".join(trigram) for trigram in entry['trigrams']])
        ])

# Generate and count global n-grams using CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 3), token_pattern=r'\b\w+\b')  # 2-gram and 3-gram
ngram_matrix = vectorizer.fit_transform(overviews)

# Extract n-grams and their counts
ngrams = vectorizer.get_feature_names_out()
counts = ngram_matrix.sum(axis=0).A1

# Combine n-grams and counts
ngram_counts = list(zip(ngrams, counts))

# Sort by frequency (optional)
sorted_ngrams = sorted(ngram_counts, key=lambda x: x[1], reverse=True)

# Print top results
print("Most common n-grams:", sorted_ngrams[:10])

# Save global bigrams and trigrams to a CSV file
with open('bigrams_trigrams.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['N-Gram', 'Frequency'])
    writer.writerows(sorted_ngrams)
