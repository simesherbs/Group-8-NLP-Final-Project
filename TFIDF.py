import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer 
from collections import defaultdict
import pickle
import math
# modified TFIDF:
# TFIDF(t, g) s.t. term t and genre g = 
    # TF(t, g ) * log (# of documents / # of docs of genre g that contain t)
                    

import os
import sys

corpus_dir = sys.argv[1]

with open(corpus_dir + '_TF_dict.pkl', 'rb') as f:
    TF = pickle.load(f)
f.close()

def calculate_tfidf(hash:dict, length):
    idf = {}
    tfidf =  defaultdict(lambda:defaultdict(float))#tfidf[id][w] = tfidf for word w in doc #id
    sentences = list(hash.values())
    id = 1
    for sent in sentences:
        doc_word_list = list(sent.keys())
        print(doc_word_list)
    return tfidf
TFIDF = calculate_tfidf(TF, 9999)
for key, value in TFIDF.items():
    print(key, value)
with open(corpus_dir + '_TFIDF_dict.pkl', 'wb') as f:
    pickle.dump(TFIDF, f)
