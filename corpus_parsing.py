import csv
import pandas as pd
import json
import sys
import os
import nltk
from nltk import PorterStemmer
from nltk.corpus import names, stopwords
from collections import defaultdict
import pickle
from nltk.tokenize import RegexpTokenizer

# Input Parsing
corpus_filename = sys.argv[1]
corpus_name_plain = ".".join(corpus_filename.split(".")[:-1])
df = pd.read_csv(corpus_filename, usecols=["overview", "genres"])
df = df.reset_index()

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
)  # genre_indexing[genre] = list of ids for documents with genre tag
processed_corpus = defaultdict(list)  # processed_corpus[id] = stemmed words of doc
idf_hash = {}  # idf_hash[word] = # of docs word appears in


ps = PorterStemmer()


def add_to_index(dict: dict, word: str, id: str):
    if word not in dict.keys():
        dict[word] = [id]
    else:
        dict[word].append(id)


def add_to_list_dict(dict: dict, keys: list[str], value: str):
    for key in keys:
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]


def add_to_idf_dict(dict: dict, tokens: list[str]):
    already_seen = {}
    for token in tokens:
        if token not in idf_hash.keys():
            idf_hash[token] = 1
        elif token not in already_seen.keys():
            idf_hash[token] += 1
            already_seen[token] = 1


for id, row in df.iterrows():

    with open(corpus_name_plain + "/" + str(id) + ".txt", "w") as f:
        tags = []

        tokenizer = RegexpTokenizer(r"\w+")
        content = row["overview"]
        tokenized = tokenizer.tokenize(str(content))
        stemmed_words = []

        for w in tokenized:
            if w.lower() not in stop_words and w not in names:
                try:
                    stemmed_word = PorterStemmer.stem(self=ps, word=w)
                    f.write(stemmed_word + " ")
                    stemmed_words.append(stemmed_word)
                    add_to_index(word_index, stemmed_word, id)
                except:
                    pass
        try:
            tags = row["genres"].split(",")
        except:
            pass
    processed_corpus[id] = stemmed_words
    add_to_idf_dict(idf_hash, stemmed_words)

    f.close()

    for tag in tags:
        if tag[0] == " ":
            tag = tag[1 : (len(tag))]
        if tag not in tag_dict:
            tag_dict[tag] = 1
            """
            os.makedirs(corpus_name_plain + "/" + tag)
            with open(corpus_name_plain + "/" + tag + "/" + str(id) + ".txt", "w") as f:
                f.write(parsed_str)
            """
        else:
            tag_dict[tag] += 1
            """
            with open(corpus_name_plain + "/" + tag + "/" + str(id) + ".txt", "w") as f:
                f.write(parsed_str)
            """
    add_to_list_dict(genre_indexing, tags, id)

sorted_by_values = dict(
    sorted(tag_dict.items(), reverse=True, key=lambda item: item[1])
)

with open(corpus_name_plain + "_genre_indexing.pkl", "wb") as fp:
    pickle.dump(genre_indexing, fp)
fp.close()

with open(corpus_name_plain + "_hash.pkl", "wb") as ff:
    pickle.dump(processed_corpus, ff)
ff.close()

with open(corpus_name_plain + "_IDF_dict.pkl", "wb") as ff:
    pickle.dump(idf_hash, ff)
ff.close()

with open(corpus_name_plain + "_word_index.pkl", "wb") as ff:
    pickle.dump(word_index, ff)
ff.close()


with open(corpus_name_plain + "_genres.json", "w") as fp:
    json.dump(sorted_by_values, fp)

with open(corpus_name_plain + "_genres.txt", "w") as fw:
    for item in sorted_by_values.items():
        fw.write(str(item) + "\n")
