from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.options.display.max_rows = 600
from pathlib import Path  
import glob
import sys


corpus_name = sys.argv[1]

dir_path  = './' + corpus_name

text_files = glob.glob(f"{dir_path}/*.txt")
text_titles = [Path(text).stem for text in text_files]

tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english', )
tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df = tfidf_df.stack().reset_index()

tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})
top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
top_tfidf[top_tfidf['term'].str.contains('zod')]


top_tfidf.to_csv('tfidf.csv')