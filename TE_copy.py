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



df = pd.read_csv("english_movies_clean.csv")
df["genres"] = df["genres"].apply(
    lambda x: x.split(", ")
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

def get_TFIDF():
    for genre, Genre_Object in Genre_Objects.items():
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(Genre_Object['df']).toarray()
        print(Genre_Object['df'])
        exit(1)
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

def get_cval(genre:str) -> DataFrame:
    cval_df = pd.read_csv('C-Value_normalized.csv')
    return cval_df[cval_df[genre]]

for genre, Genre_Object in Genre_Objects.items():
    atfidf_df = Genre_Object['atfidf']
    chi2_df = Genre_Object['chi_square']
    print(atfidf_df.shape)
    print(chi2_df.shape)
    # Normalize TF-IDF scores
    atfidf_df['normalized_score'] = (atfidf_df['score'] - atfidf_df['score'].min()) / (atfidf_df['score'].max() - atfidf_df['score'].min())
    chi2_df['normalized_score'] = (chi2_df['score'] - chi2_df['score'].min()) / (chi2_df['score'].max() - chi2_df['score'].min())


    atfidf_df = atfidf_df.sort_values(by='term')

    print(atfidf_df.shape)

    scores = pd.DataFrame({
        'term': atfidf_df['term'],
        'CS': (.3 * atfidf_df['normalized_score']) + (.3 * chi2_df['normalized_score'] )
    }).sort_values(by='CS', ascending=False).head(100)

    Genre_Object['CS'] = scores['term'].tolist()


with open('CT.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['genre', 'terms'])
    for genre, Genre_Object in Genre_Objects.items():
        terms  = Genre_Object['CS']
        term_str = ''
        for term in terms:
            term_str += term + ' '
        writer.writerow([genre, term_str])