import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv
import nltk
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk import PorterStemmer

ps = PorterStemmer()
# Load the movie corpus
corpus_file = "english_movies.csv"  # Replace with your CSV file path
df = pd.read_csv(corpus_file)

# Ensure the 'overview' column exists and drop any rows where it's missing
if 'overview' not in df.columns:
    raise ValueError("The CSV does not have an 'overview' column.")
df = df.dropna(subset=['overview'])

genres = df['genres'].tolist()

stop_words = set(stopwords.words("english"))

# Extract the overview column
overviews = df['overview'].tolist()

tokenizer = RegexpTokenizer(r"[A-z][A-z-]*(?:'s)?")

def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    :param text: A string (sentence or document).
    :param n: The n in n-gram (e.g., 2 for bigram, 3 for trigram).
    :return: A list of n-grams as tuples.
    """
    sent_text = nltk.sent_tokenize(text)
    ngrams = ''
    for sentence in sent_text:
        tokens = tokenizer.tokenize(sentence)  # Split the text into words
        for i in range(len(tokens) -n+1):
            ngram = tokens[i:i+n]
            ngram_stemmed = ''
            if (ngram[0] not in stop_words and ngram[n-1] not in stop_words):
                    for j in range(0, n):
                        ngram_stemmed += (PorterStemmer.stem(self=ps, word=ngram[j].lower()))
                        if j < n-1:
                            ngram_stemmed += ' '
                    if ngrams == '':
                        ngrams += str(ngram_stemmed)
                    else:
                        ngrams += '@@' + str(ngram_stemmed)
    return ngrams

def clean_unagram(overview):

    unagrams = ''
    tokens = tokenizer.tokenize(overview)
    for token in tokens:
        if token not in stop_words:
            if unagrams == '':
                unagrams += PorterStemmer.stem(self=ps, word=token.lower())
            else:
                unagrams += '@@' + PorterStemmer.stem(self=ps, word=token.lower())
    return unagrams
# Generate bigrams and trigrams for each overview and keep results per entry
entries_bigrams_trigrams = []

for i, overview in enumerate(overviews):
    bigrams = generate_ngrams(overview, 2)
    trigrams = generate_ngrams(overview, 3)
    ungrams = clean_unagram(overview)
    entries_bigrams_trigrams.append({
        "index": i,
        "overview": ungrams + '@@' + bigrams + '@@' + trigrams,
        "genres": genres[i]
    })

# Save the bigrams and trigrams for each entry to a CSV file
with open('entry_bigrams_trigrams_v2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'overview', 'genres'])
    for entry in entries_bigrams_trigrams:
        writer.writerow([
            entry['index'],
            entry['overview'],
            entry['genres']
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
