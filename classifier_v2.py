import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk import PorterStemmer
import math
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r"[A-z][A-z-]*(?:'s)?")

CS_dict = pd.read_csv('CS_dict.csv')
CS_dict = pd.DataFrame(CS_dict).set_index('term')

genres_arr = sorted([
    "Drama",
    "Comedy",
    "Thriller",
    "Action",
    "Adventure",
    "Horror",
    "Romance",
    "Family",
    "Crime",
    "Science Fiction",
    "Fantasy",
    "Animation",
    "Mystery",
    "History",
    "Music",
    "TV Movie",
    "War",
    "Documentary",
    "Western",
])
def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    :param text: A string (sentence or document).
    :param n: The n in n-gram (e.g., 2 for bigram, 3 for trigram).
    :return: A list of n-grams as tuples.
    """
    sent_text = nltk.sent_tokenize(str(text))
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
    tokens = tokenizer.tokenize(str(overview))
    for token in tokens:
        if token not in stop_words:
            if unagrams == '':
                unagrams += PorterStemmer.stem(self=ps, word=token.lower())
            else:
                unagrams += '@@' + PorterStemmer.stem(self=ps, word=token.lower())
    return unagrams
"""
    Split training and test set. 

        Calculate feature score for training set

        fit model

    "look up" CS for test set

        run model

    
"""


"""
    ANOTHER IDEA:

        for the train set, split every entry into a single genre
            
            calculate CS score for each word and then mutliply together for get CS for genre so like:


                ID   Action CS      Adventure CS        ..              tags
                1       .7                 .5                           [Action...]

            OR

            Maybe average them?

"""


test = "When his attempt to save his family inadvertently alters the future, Barry Allen becomes trapped in a reality in which General Zod has returned and there are no Super Heroes to turn to. In order to save the world that he is in and return to the future that he knows, Barry's only hope is to race for his life. But will making the ultimate sacrifice be enough to reset the universe?"


def calculate_overview_CS(overview:str, index:int):
    tokens = clean_unagram(overview) + '@@' + generate_ngrams(overview, 2) + '@@' + generate_ngrams(overview, 3)
    tokens = tokens.split('@@')
    scores = np.zeros(len(genres_arr))
    score = 0
    for token in tokens:
        try:
            row = CS_dict.loc[token].to_numpy()
            scores += np.nan_to_num(row, 0)
        except:
            print(index)
            pass
        
    return scores


corpus = pd.read_csv('english_movies.csv')
corpus["genres"] = corpus["genres"].apply(
    lambda x: str(x).split(", ")
)
X = pd.DataFrame(columns= genres_arr + ['tags'])

for i, row in corpus.iterrows():
    X_row = [i]
    scores = calculate_overview_CS(row['overview'], i)
    X.loc[i] = scores.tolist() + [row['genres']]
X.to_csv('CS_by_overview.csv')
