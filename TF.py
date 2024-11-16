import pickle 
import os
import sys
from collections import defaultdict


corpus_dir = sys.argv[1]
rootdir = os.curdir + "/" + corpus_dir



def add_to_dict(dict:dict, key:str):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


#tfidf[g][w] = tfidf for word w in grene g
TF = {}
for path, folders, files in os.walk(rootdir):
    for file in files:
        file_dict = {}
        with open(rootdir + "/" + file, "r") as f:
            content = f.read().split(" ")
            for word in content:
                add_to_dict(file_dict, word)
        f.close()
        TF[file] = file_dict

with open(corpus_dir + '_TF_dict.pkl', 'wb') as f:
    pickle.dump(TF, f)

        
