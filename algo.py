import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import names, stopwords
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
svm = LogisticRegression()
clf = OneVsRestClassifier(svm)
stop_words = set(stopwords.words("english"))



def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)

    return multilabel_binarizer.inverse_transform(q_pred)
def clean_text(text):
    
    
    text = re.sub("\'", "", text) 
   
    text = re.sub("[^a-zA-Z]"," ",text) 
 
    text = ' '.join(text.split()) 

    text = text.lower() 
    
    return text
pd.set_option('display.max_colwidth', 300)
corpus_name = sys.argv[1]

movies = pd.read_csv(corpus_name + '_clean.csv')

multilabel_binarizer = MultiLabelBinarizer()

genres = []
for i in movies["genres"]:
    genres.append(str(i).split(", "))


movies['cleaned_genres'] = genres
multilabel_binarizer.fit(movies['cleaned_genres'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)



y = multilabel_binarizer.transform(movies['cleaned_genres'])

print(multilabel_binarizer.classes_)
print(y)

xtrain, xval, ytrain, yval = train_test_split(movies['overview'], y, test_size=0.5, random_state=42)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)



clf.fit(xtrain_tfidf, ytrain)

y_pred_prob = clf.predict_proba(xval_tfidf)

threshold = 0.3 
y_pred_new = (y_pred_prob >= threshold).astype(int)

a = f1_score(yval, y_pred_new, average="micro")

print(a)

for i in range(5): 
  k = xval.sample(1).index[0] 
  print("Movie: ", movies['id'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies['cleaned_genres'][k], "\n")