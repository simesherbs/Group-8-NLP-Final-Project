from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd
from nltk import PorterStemmer, sent_tokenize, word_tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
# Load the movie corpus
corpus_file = "english_movies.csv"  # Replace with your CSV file path
df = pd.read_csv(corpus_file)

# Ensure the 'overview' column exists and drop any rows where it's missing
if 'overview' not in df.columns:
    raise ValueError("The CSV does not have an 'overview' column.")
df = df.dropna(subset=['overview'])

# Extract the overview column
overviews = df['overview'].tolist()

punctuation_list = ['.', ',', '"', ":", "-", "--", ";", ".", "?", "!"]

def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    :param text: A string (sentence or document).
    :param n: The n in n-gram (e.g., 2 for bigram, 3 for trigram).
    :return: A list of n-grams as tuples.
    """
    tokens = text.split()  # Split the text into words
    ngrams = []
    sent_text = sent_tokenize(text=text)
    for sentence in sent_text:
        tokens = word_tokenize(sentence)
        for i in range(len(tokens)-n+1):
            num_of_stop_words = 0
            stemmed = []
            ngram = tokens[i:i+n]
            not_punc = True
            for unigram in ngram:
                if unigram in punctuation_list:
                    not_punc = False
                    break
                if unigram.lower() in stop_words:
                    num_of_stop_words += 1
                    stemmed.append(unigram.lower())
                else:
                    stemmed.append(PorterStemmer.stem(self=ps, word=str(unigram).lower()))
            if (num_of_stop_words < n and not_punc):
                ngrams.append(tuple(stemmed))
    return ngrams

# Generate bigrams and trigrams for each overview and keep results per entry
entries_bigrams_trigrams = []

for i, overview in enumerate(overviews):
    unigrams = generate_ngrams(overview, 1)
    bigrams = generate_ngrams(overview, 2)
    trigrams = generate_ngrams(overview, 3)
    entries_bigrams_trigrams.append({
        "index": i,
        "overview": overview,
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams
    })

# Save the bigrams and trigrams for each entry to a CSV file
with open('entry_bigrams_trigrams.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Overview', 'Unigrams', 'Bigrams', 'Trigrams'])
    for entry in entries_bigrams_trigrams:
        writer.writerow([
            entry['index'],
            entry['overview'],
            ", ".join([" ".join(unigram) for unigram in entry['unigrams']]),
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
