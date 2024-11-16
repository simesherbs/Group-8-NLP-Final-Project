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
with open(corpus_dir + '_hash.pkl', 'rb') as fa:
    hash = pickle.load(fa)
fa.close()
with open(corpus_dir + '_IDF_dict.pkl', 'rb') as fa:
    IDF = pickle.load(fa)
fa.close()


def calculate_tfidf(hash:dict, length):
    tfidf = {}
    for id, tokens in hash.items():
        current_id_hash = {}
        for token in tokens:
            
            tf = TF[id][token]
    
           
            idf = math.log(length/IDF[token])

            current_id_hash[token] = tf*idf
        tfidf[id] = current_id_hash
 
    return tfidf
TFIDF = calculate_tfidf(hash, 9999)

with open(corpus_dir + '_TFIDF_dict.pkl', 'wb') as f:
    pickle.dump(TFIDF, f)
