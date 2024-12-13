import pandas as pd
from collections import defaultdict, Counter
import ast  # To safely evaluate stringified lists in the CSV

def find_nested_terms_and_frequencies_from_csv(csv_file):
    """
    Identifies nested terms and their frequencies from a CSV file with document-level n-grams and genres.

    Args:
        csv_file (str): Path to the CSV file with columns 'unigrams', 'bigrams', 'trigrams', and 'genres'.

    Returns:
        dict: A dictionary where keys are genres, and values are dictionaries mapping terms
              to a tuple of (nested terms, frequency).
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Initialize the nested terms and frequencies data structure
    genre_term_data = defaultdict(lambda: defaultdict(lambda: {"nested_terms": [], "frequency": 0}))

    # Iterate over each row/document in the CSV
    for _, row in df.iterrows():
        # Parse the columns
        unigrams = str(row['Unigrams']).split(', ')
        bigrams = str(row['Bigrams']).split(', ')
        trigrams = str(row['Trigrams']).split(', ')
        genres = str(row['Genres']).split(', ')
        

        # Combine n-grams for this document
        doc_ngrams = unigrams + bigrams + trigrams

        # Count term frequencies for the document
        term_freq = Counter(doc_ngrams)

        # Compare all terms within the document
        for term in doc_ngrams:
            for other_term in doc_ngrams:
                if term != other_term and term in other_term:
                    #print(term, other_term)
                   
                    for genre in genre:
                        
                        genre_term_data[genre][term]["nested_terms"].append(other_term)
                        #print(genre_term_data[genre][term]["nested_terms"])
                        
            # Update term frequency
            for genre in genres:
                genre_term_data[genre][term]["frequency"] += term_freq[term]

    return genre_term_data

# Example Usage
csv_file = "entry_bigrams_trigrams.csv"  # Replace with your CSV file path

nested_terms_with_freq = find_nested_terms_and_frequencies_from_csv(csv_file)

# Print the output for the first genre
print (nested_terms_with_freq['Action']['famili'])
for genre, terms in nested_terms_with_freq.items():
    print(f"Genre: {genre}")
    for term, data in terms.items():
        if len(data['nested_terms']) != 0:
            print(f"  Term: {term}")
            print(f"    Frequency: {data['frequency']}")
            print(f"    Nested Terms: {data['nested_terms']}")
    break  # Print just the first genre for clarity
