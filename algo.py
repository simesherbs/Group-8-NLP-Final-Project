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
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
svm = SVC(probability=False, C=100)
rf_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

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
Y_genres = multilabel_binarizer.fit_transform(movies['cleaned_genres'])
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(movies['overview'])

# Step 3: Train Random Forest Classifier
rf_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_classifier.fit(X_tfidf, Y_genres)

# Step 4: Extract Feature Importances
# Average feature importance across all classes
feature_importances = np.mean([
    estimator.feature_importances_ for estimator in rf_classifier.estimators_
], axis=0)

# Map feature importance to terms
terms = vectorizer.get_feature_names_out()
importance_df = pd.DataFrame({
    "Term": terms,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Top N terms
top_n_terms = importance_df.head(10)
print("Top Terms Based on Feature Importance:\n", top_n_terms)