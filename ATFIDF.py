import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer 
from collections import defaultdict
import pickle
import math
import sys
import os

corpus_dir = sys.argv[1]
with open(corpus_dir + '_TFIDF_dict.pkl', 'rb') as f:
    TFIDF = pickle.load(f)

print(TFIDF)