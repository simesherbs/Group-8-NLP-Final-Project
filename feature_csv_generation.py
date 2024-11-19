import pickle
import sys
import csv

corpus_dir = sys.argv[1]

with open(corpus_dir + "_ATFIDF_dict.pkl", "rb") as f:
    ATFIDF = dict(pickle.load(f))
f.close()

genres = [
    "Drama",
    "Comedy",
    "Thriller",
    "Action",
    "Adventure",
    "Horror",
    "Romance",
    "Family",
    "Crime",
    "Science Fiction",
    "Fantasy",
    "Animation",
    "Mystery",
    "History",
    "Music",
    "TV Movie",
    "War",
    "Documentary",
    "Western",
]


fields = ["word", "ATFIDF_score"]

with open(corpus_dir + "_features.csv", "w", newline="") as file:
    for word, scores in ATFIDF.items():
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
file.close()
