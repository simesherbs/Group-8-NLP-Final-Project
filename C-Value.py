import pandas as pd
from collections import defaultdict, Counter
import ast  # To safely evaluate stringified lists in the CSV
import math
import statistics

N = 5
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
    genre_term_freq = defaultdict(Counter)

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
                   
                    for genre in genres:
                        
                        genre_term_data[genre][term]["nested_terms"].append(other_term)
                        #print(genre_term_data[genre][term]["nested_terms"])
                        
            # Update term frequency
            for genre in genres:
                genre_term_data[genre][term]["frequency"] += term_freq[term]
    return genre_term_data

def get_nested_freq_sum(nested_terms, main_dict, genre):
    sum = 0
    for term in nested_terms:
        sum += main_dict[genre][term]['frequency']
    return sum
def compute_c_value(candidate_str, genre_freq, nested_terms, main_dict, genre):
    """
    Computes the C-Value for a single term.
    """

    if len(nested_terms) != 0:
        return math.log2(len(candidate_str))* (genre_freq  - (1/len(nested_terms) * get_nested_freq_sum(nested_terms, main_dict, genre)))
    return (math.log2(len(candidate_str))*genre_freq)   


# Example Usage
csv_file = "entry_bigrams_trigrams.csv"  # Replace with your CSV file path

nested_terms_with_freq = find_nested_terms_and_frequencies_from_csv(csv_file)

# Print the output for the first genre
#print (nested_terms_with_freq)
print(nested_terms_with_freq['Action']['famili']['nested_terms'])
print(nested_terms_with_freq['Action']['famili']['frequency'])
print(nested_terms_with_freq['Action']['famili inadvert']['frequency'])
print(nested_terms_with_freq['Action']['famili inadvert alter']['frequency'])

c_value_dict = defaultdict(lambda: defaultdict(float))

for genre, terms in nested_terms_with_freq.items():
    for term, data in terms.items():
        c_value_dict[genre][term] = compute_c_value(term, data['frequency'], data['nested_terms'], nested_terms_with_freq, genre)
        #print(f"  Term: {term}")
        #print(f"    C-Value: {compute_c_value(term, data['frequency'], data['nested_terms'], nested_terms_with_freq, genre)}")
        #print(f"    Nested Terms: {data['nested_terms']}")
      # Print just the first genre for clarity
cval = pd.DataFrame.from_dict(c_value_dict)
genres = cval.columns
cval.rename(columns={'': 'term'}, inplace=True)
cval.fillna(0, inplace=True)
cval = cval.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
cval["variance"] = cval.iloc[:, 1:].var(axis=1)
cval["mean"] = cval.iloc[:, 1:].mean(axis=1)
cval["variance_to_mean_ratio"] = cval["variance"] / cval["mean"]
filtered = cval[cval[genres].gt(10.0).any(axis=1)]
filtered.sort_values(by='variance_to_mean_ratio', ascending=False).to_csv('C-Value.csv')
print(filtered.sort_values(by='variance_to_mean_ratio', ascending=False))