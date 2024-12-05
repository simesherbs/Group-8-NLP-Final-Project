import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer 
from collections import defaultdict
import pickle
import math
import sys
import os
import statistics
corpus_dir = sys.argv[1]
with open(corpus_dir + '_TFIDF_dict.pkl', 'rb') as f:
    TFIDF = pickle.load(f)
f.close()

with open(corpus_dir + '_word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)
f.close()

with open(corpus_dir + '_genre_indexing.pkl', 'rb') as f:
    genre_indexing = pickle.load(f)
f.close()


def get_genres_of_id(id:int):
    genres = []
    for genre, id_list in genre_indexing.items():
        if id in id_list:
            genres.append(genre)
    return genres

genre_list = list(genre_indexing.keys())
ATFIDF = {}


for word, ids in word_index.items():
    cur_word_hash = {}
    for id in ids:
        tfidf = TFIDF[id][word]
        genres = get_genres_of_id(id)
        for genre in genres:
            if genre not in cur_word_hash.keys():
                cur_word_hash[genre] = [tfidf]
            else:
                cur_word_hash[genre].append(tfidf)
    for genre, vals in cur_word_hash.items():
        cur_word_hash[genre] = statistics.fmean(vals)
    ATFIDF[word] = cur_word_hash



with open(corpus_dir + '_ATFIDF_dict.pkl', 'wb') as f:
    pickle.dump(ATFIDF, f)
f.close()
