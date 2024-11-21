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

def create_file(id: str, words: list[str], genres: list[str]):
    
    with open(corpus_name_plain + "/" + str(id) + ".txt", "w") as f:
        for word in words:
            try:
                word.encode("utf-8")
                f.write(word + " ")
            except Exception as e:
                print(word, id)
                exit(1)
    f.close()

def append_doc_freq(id: str, words: list[str], genres: list[str]):
    already_seen = {}
    for word in words:
        already_seen_keys = already_seen.keys()
        doc_freq_keys = doc_freq_keys.keys()
        if word not in doc_freq.keys():
            doc_freq[word] = 1
        elif word not in already_seen.keys():
            doc_freq[word] += 1
            already_seen[word] = 1

def append_word_index(id: str, words: list[str], genres: list[str]):
    for word in words:
        keys = word_index.keys()
        if word not in word_index.keys():
            word_index[word] = [id]
        else:
            word_index[word].append(id)

def append_genre_index(id: str, words:list[str], genres: list[str]):
    genre_indexing[id] = []
    for genre in genres:
        if genre[0] == " ":
            genre = genre[1:(len(genre))]
            genre_indexing[id].append(genre)

def process_corpus(id:str, words:list[str], genres: list[str]):
    processed_corpus[id] = words

def parse_row(id: str, row: Series):
    content = row["overview"]
    tokenized = tokenizer.tokenize(str(content))
    stemmed_words = []
    for w in tokenized:
        if w.lower() not in stop_words and w not in names:
            stemmed_word = PorterStemmer.stem(self=ps, word=w)
            stemmed_words.append(stemmed_word)
    genres = (str(row["genres"]).split(","))

    return (id, stemmed_words, genres)

# External data
stop_words = set(stopwords.words("english"))
male_names = names.words("male.txt")
female_names = names = names.words("female.txt")
names = female_names + male_names

# Processing structures
word_index = {}  # word_index[word] = list of documents word appears in
tag_dict = {}  # tag_dict[tag] = # of movies with tag
genre_indexing = defaultdict(
    list
)  # genre_indexing[id] = list of genres for document id
processed_corpus = defaultdict(list)  # processed_corpus[id] = list of stemmed words of doc
doc_freq = {}  # doc_freq[word] = # of docs word appears in

# Input Parsing
corpus_filename = sys.argv[1]
opts, args  = getopt.getopt(args=sys.argv[2:], shortopts="",longopts=['dir_creation', 'doc_freq', 'word_index', 'genre_index', 'process_corpus']) 
opt_dict = {
    '--dir_creation': (create_file, None),
    '--doc_freq': (append_doc_freq, doc_freq),
    '--word_index': (append_word_index, word_index),
    '--genre_index': (append_genre_index, genre_indexing),
    '--process_corpus': (process_corpus, processed_corpus),
}
f_array = []
d_dict = {}
for opt, arg in opts:
    if opt in opt_dict:
        f_array.append(opt_dict[opt][0])
        if opt_dict[opt][1] is not None: d_dict[opt[2:]] = opt_dict[opt][1]
df = pd.read_csv(corpus_filename, usecols=["overview", "genres"])
df = df.reset_index()
corpus_name_plain = ".".join(corpus_filename.split(".")[:-1])



ps = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+")


for id, row in df.iterrows():

    id, content, genres = parse_row(id, row)
    for func in f_array:
        func(id, content, genres)

for d_name, d in d_dict.items():
    with open(corpus_name_plain + '_' + d_name + '.pkl', 'wb') as f:
        pickle.dump(d, f)
    f.close()
