import pandas as pd

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
    :return: A list of n-grams as strings.
    """
    tokens = text.split()  # Split the text into words
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Generate bigrams and trigrams for each overview
bigrams_column = []
trigrams_column = []

for overview in overviews:
    bigrams = generate_ngrams(overview, 2)
    trigrams = generate_ngrams(overview, 3)

    bigrams_column.append(", ".join(bigrams))
    trigrams_column.append(", ".join(trigrams))

# Add bigrams and trigrams as new columns to the DataFrame
df['bigrams'] = bigrams_column
df['trigrams'] = trigrams_column

# Save the updated DataFrame to a new CSV file
updated_file = "english_movies_with_ngrams.csv"
df.to_csv(updated_file, index=False)

print(f"Updated CSV saved to {updated_file}")