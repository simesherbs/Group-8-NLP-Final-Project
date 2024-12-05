import pickle 
import os
import sys
from collections import defaultdict


corpus_dir = sys.argv[1]
rootdir = os.curdir + "/" + corpus_dir

with open(corpus_dir + '_hash.pkl', 'rb') as f:
    hash = pickle.load(f)
f.close()

def add_to_dict(dict:dict, key:str):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1



TF = {}
for file, tokens in hash.items():
    current_file_dict = {}
    for token in tokens:
        add_to_dict(current_file_dict, token)
    TF[file] = current_file_dict


with open(corpus_dir + '_TF_dict.pkl', 'wb') as f:
    pickle.dump(TF, f)

        
