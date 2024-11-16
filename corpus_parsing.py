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

def add_to_dict(dict:dict, key:str, value: str):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]

for index, row in df.iterrows():
    
    title = row['title']
    parsed_str = ""
    with open(corpus_name_plain + "/" + str(index) + ".txt", "w") as f:
        tags = []

        
        content = row['overview']
        tokenized = nltk.tokenize.word_tokenize(str(content))

        
        for w in tokenized:
            if w.lower() not in stop_words and w not in names:
                try:
                    f.write(PorterStemmer.stem(self=ps, word=w) + " ")
                    parsed_str += (w + " ")
                except:
                    pass
        try:
            tags = row['genres'].split(',')
        except:
            print(title)
      
      
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
    add_to_dict(genre_indexing, tag, index)

sorted_by_values = dict(sorted(tag_dict.items(), reverse=True, key=lambda item: item[1]))

with open(corpus_name_plain + '_genre_indexing.pkl', 'wb') as fp:
    pickle.dump(genre_indexing, fp)
fp.close()

with open(corpus_name_plain + '_genres.json', 'w') as fp:
    json.dump(sorted_by_values, fp)

with open(corpus_name_plain + '_genres.txt', 'w') as fw:
    for item in sorted_by_values.items():
        fw.write(str(item) + "\n")

df = df.reset_index()  # make sure indexes pair with number of rows

#for index, row in df.iterrows():

