from typing import TypedDict
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
import csv



df = pd.read_csv("entry_bigrams_trigrams_v2.csv")
df["genres"] = df["genres"].apply(
    lambda x: str(x).split(", ")
)  # Assuming genres are comma-separated


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

class Genre(TypedDict):
    genre: str
    tfidf_scores: DataFrame
    atfidf: DataFrame
    chi_square: DataFrame
    df: DataFrame
    terms: ndarray
    CS: list
    CS_dict: dict

Genre_Objects = {}
CT = []
class Word(TypedDict):
    word: str
    c_value: float
    tfidf: dict[str, float]
    chi_square: dict[str, float]
    frequency: dict[str, float]

for genre in genres_arr:
    gdf = df[df['genres'].apply(lambda x: genre in x)]
    
    Genre_Objects[genre] = Genre(df=gdf['overview'], genre=genre)
    
def tokenizer(text):
    temp = text.split('@@')
    return temp
def get_TFIDF():
    for genre, Genre_Object in Genre_Objects.items():
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
        X = vectorizer.fit_transform(Genre_Object['df']).toarray()
        terms = vectorizer.get_feature_names_out()
        terms = sorted(list(set(terms)))
        Genre_Object['terms'] = terms
        CT.extend(terms)
    
        # Display the shape of the original TF-IDF sparse matrix

        atfidf = np.mean(X, axis=0)
        tfidf_df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
        Genre_Object['tfidf_scores'] = tfidf_df
        Genre_Object['atfidf'] = pd.DataFrame({
            'term': terms,
            'score': atfidf
        }).sort_values(by='term')
get_TFIDF()
CT = sorted(list(set(CT)))
def get_chi_square():

    # X needs to be like:
    """
                advice assist attempt  
        Doc 1      1      0      1          
        DOc 2      0      0      1          
    
    y needs to be like:

        [1, 0] for <current genre>, meaning:
            Doc1 is in genre, Doc2 is not   
    """
    
    vectorizer = CountVectorizer(vocabulary=CT, ngram_range=(1,3))
    X = vectorizer.fit_transform(df['overview'])
    for genre, Genre_Object in Genre_Objects.items():
        a = df['genres'].apply(lambda genre_list: 1 if genre in genre_list else 0)
        chi2_scores, p_values = chi2(X, a)
        chi2_results = pd.DataFrame({
            'term': CT,
            'score': chi2_scores,
            'p_value': p_values
        }).sort_values(by='term')
        Genre_Object['chi_square'] = chi2_results[chi2_results['term'].isin(Genre_Object['terms'])].reset_index(drop=True)
        
get_chi_square()



for genre, Genre_Object in Genre_Objects.items():
    atfidf_df = Genre_Object['atfidf']
    chi2_df = Genre_Object['chi_square']
    # Normalize TF-IDF scores
    atfidf_df['normalized_score'] = (atfidf_df['score'] - atfidf_df['score'].min()) / (atfidf_df['score'].max() - atfidf_df['score'].min())
    chi2_df['normalized_score'] = (chi2_df['score'] - chi2_df['score'].min()) / (chi2_df['score'].max() - chi2_df['score'].min())

    atfidf_df = atfidf_df.sort_values(by='term')

    

    scores = pd.DataFrame({
        'term': atfidf_df['term'],
        'CS': (.3 * atfidf_df['normalized_score']) + (.7 * chi2_df['normalized_score'])
    }).sort_values(by='term', ascending=False)
    Genre_Object['CS_dict'] = dict(zip(scores.term, scores.CS))

    aaa = scores.sort_values(by='CS', ascending=False).head(10)
    Genre_Object['CS'] = aaa['term'].tolist()


with open('CT_withngrams.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['genre', 'terms'])
    for genre, Genre_Object in Genre_Objects.items():
        terms  = Genre_Object['CS']
        term_str = ''
        for term in terms:
            term_str += term + ', '
        writer.writerow([genre, term_str])

with open('CS_dict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['term'] + genres_arr)
    for term in CT:
        row = [term]
        for genre, Genre_Object in Genre_Objects.items():
            if term in Genre_Object['CS_dict']:
                row.append(Genre_Object['CS_dict'][term])
            else:
                row.append(0)
        writer.writerow(row)
            

"""
        Action CT score     Adventure CT score
word1       .8                  .9
word2       
word3





"""