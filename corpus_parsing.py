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

word_index = {}

corpus_filename = sys.argv[1]
corpus_name_plain = '.'.join(corpus_filename.split('.')[:-1])

stop_words = set(stopwords.words('english'))
male_names = names.words('male.txt')
female_names = names = names.words('female.txt')
names = female_names + male_names
tag_dict = {}

df = pd.read_csv(corpus_filename, usecols=['title', 'overview', 'genres'])
df = df.reset_index()  # make sure indexes pair with number of rows

ps = PorterStemmer()

genre_indexing  = defaultdict(list)

hash = defaultdict(list)

idf_hash = {}

def add_to_index(dict:dict, word:str, id:str):
    if word not in dict.keys():
        dict[word] = [id]
    else:
        dict[word].append(id)

def add_to_list_dict(dict:dict, keys:list[str], value: str):
    for key in keys:
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]

def add_to_idf_dict(dict:dict, tokens:list[str]):
    already_seen = {}
    for token in tokens:
        if token not in idf_hash.keys():
            idf_hash[token] = 1
        elif token not in already_seen.keys():
            idf_hash[token] += 1
            already_seen[token] = 1
    

for index, row in df.iterrows():
    
    title = row['title']
    parsed_str = ""
    with open(corpus_name_plain + "/" + str(index) + ".txt", "w") as f:
        tags = []

        tokenizer = RegexpTokenizer(r'\w+')
        content = row['overview']
        tokenized = tokenizer.tokenize(str(content))
        stemmed_words = []
        
        for w in tokenized:
            if w.lower() not in stop_words and w not in names:
                try:
                    f.write(PorterStemmer.stem(self=ps, word=w) + " ")
                    parsed_str += (w + " ")
                    stemmed_words.append(PorterStemmer.stem(self=ps, word=w))
                    add_to_index(word_index, PorterStemmer.stem(self=ps, word=w), index)
                except:
                    pass
        try:
            tags = row['genres'].split(',')
        except:
            print(title)
    hash[index] = stemmed_words
    add_to_idf_dict(idf_hash, stemmed_words)
      
    f.close()
    
    for tag in tags:
        if tag[0] == " ":
            tag = tag[1:(len(tag))]
        if tag not in tag_dict:
            tag_dict[tag] = 1
            """
            os.makedirs(corpus_name_plain + "/" + tag)
            with open(corpus_name_plain + "/" + tag + "/" + str(index) + ".txt", "w") as f:
                f.write(parsed_str)
            """
        else:
            tag_dict[tag] += 1
            """
            with open(corpus_name_plain + "/" + tag + "/" + str(index) + ".txt", "w") as f:
                f.write(parsed_str)
            """
    add_to_list_dict(genre_indexing, tags, index)

sorted_by_values = dict(sorted(tag_dict.items(), reverse=True, key=lambda item: item[1]))

with open(corpus_name_plain + '_genre_indexing.pkl', 'wb') as fp:
    pickle.dump(genre_indexing, fp)
fp.close()

with open(corpus_name_plain + '_hash.pkl', 'wb') as ff:
    pickle.dump(hash, ff)
ff.close()




with open(corpus_name_plain + '_IDF_dict.pkl', 'wb') as ff:
    pickle.dump(idf_hash, ff)
ff.close()

with open(corpus_name_plain + '_word_index.pkl', 'wb') as ff:
    pickle.dump(word_index, ff)
ff.close()


with open(corpus_name_plain + '_genres.json', 'w') as fp:
    json.dump(sorted_by_values, fp)

with open(corpus_name_plain + '_genres.txt', 'w') as fw:
    for item in sorted_by_values.items():
        fw.write(str(item) + "\n")

df = df.reset_index()  # make sure indexes pair with number of rows

#for index, row in df.iterrows():

