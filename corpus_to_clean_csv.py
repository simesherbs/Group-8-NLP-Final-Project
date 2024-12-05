import csv
import pandas as pd
from pandas import Series
import json
import sys
import os
import getopt
import nltk
from nltk import PorterStemmer
from nltk.corpus import names, stopwords
from collections import defaultdict
import pickle
from nltk.tokenize import RegexpTokenizer
import re
corpus_name = sys.argv[1]

directory = os.fsencode('./' + corpus_name)
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words("english"))
male_names = names.words("male.txt")
female_names = names = names.words("female.txt")
names = female_names + male_names

def parse_row(id: str, row: pd.Series):
    content = row["Overview"]
    tokenized = tokenizer.tokenize(str(content))
    stemmed_words = ''
    stemmed_bigrams = str(row['Bigrams']).split(', ')
    stemmed_trigrams = str(row['Trigrams']).split(', ')

    for w in tokenized:
        if w.lower() not in stop_words and w not in names:
            stemmed_word = PorterStemmer.stem(self=ps, word=w)
            stemmed_words += stemmed_word + '@@'
    for token in stemmed_bigrams:
        stemmed_words += token + '@@'
    for token in stemmed_trigrams:
        stemmed_words += token + '@@'
    genres = row['Genres']

    return [id, stemmed_words, genres]

csv_outfile = open(corpus_name + '_clean.csv', "w")
df = pd.read_csv(corpus_name + '.csv', usecols=["Overview", "Bigrams", "Trigrams", "Genres"])
writer = csv.writer(csv_outfile, delimiter=',')
writer.writerow(['id', 'overview', 'genres'])
for id, row in df.iterrows():
    writer.writerow(parse_row(id, row))

