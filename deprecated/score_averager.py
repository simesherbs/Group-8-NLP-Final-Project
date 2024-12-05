from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pandas import DataFrame
pd.options.display.max_rows = 5
from pathlib import Path  
import glob
import sys
import getopt
import csv
import pickle



def averager(infile: str, outfile:str):
    
    csv_file = open(outfile + '.csv', "w")
    writer = csv.writer(csv_file, delimiter=',')
    df = pd.read_csv(infile + ".csv")
    word_list = df['term'].tolist()
    writer.writerow(['term'] + genre_list)
    new_df = DataFrame(columns=['term'] + genre_list)
    add_genre_tags(df)

    for word in word_list:
        err = 0
        row_arr = [word]
        word_df = df.loc[df['term'] == word]

        for genre in genre_list:
            genre_mask = word_df['genres'].apply(lambda x: genre in x)
            genre_df = word_df[genre_mask]
            try:
                row_arr.append(genre_df['tfidf'].mean())
            except:
                pass
        if err == 0:
            writer.writerow(row_arr)
def add_genre_tags(df: DataFrame):
    df['genres'] = ''
    for index, row in df.iterrows():
        doc_id = df.at[index, 'document']
        df.at[index, 'genres'] = genre_index[doc_id]
    
    

genre_list = [

    'Drama',
    'Comedy',
    'Thriller',
    'Action',
    'Adventure',
    'Horror',
    'Romance',
    'Family',
    'Crime',
    'Science Fiction',
    'Fantasy',
    'Animation',
    'Mystery',
    'History',
    'Music',
    'TV Movie',
    'War',
    'Documentary',
    'Western'
]



corpus = sys.argv[1]
with open(corpus + '_genre_index.pkl', 'rb') as f:
    genre_index = pickle.load(f)
f.close()
with open(corpus + '_word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)
f.close()

#word_list = list(word_index.keys())

opts, args  = getopt.getopt(args=sys.argv[2:], shortopts="",longopts=['tfidf']) 
opt_dict = {
    '--tfidf': 'tfidf'
}
d_array = []
for opt, arg in opts:
    if opt in opt_dict:
        d_array.append(opt_dict[opt])
for d in d_array:
    averager(d, 'averaged_' + d)

