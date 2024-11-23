from typing import TypedDict
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from numpy import ndarray

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
    chi_square: DataFrame
    df: DataFrame
    terms: ndarray

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
        terms = vectorizer.get_feature_names_out()
        Genre_Object['terms'] = terms
        CT.extend(terms.tolist())
        # Display the shape of the original TF-IDF sparse matrix
        tfidf_df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
        Genre_Object['tfidf_scores'] = tfidf_df
get_TFIDF()
CT = list(set(CT))
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
            'chi2_score': chi2_scores,
            'p_value': p_values
        }).sort_values(by='term')
        Genre_Object['chi_square'] = chi2_results

get_chi_square()

