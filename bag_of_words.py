import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from bigramstrigrams import generate_ngram_file
from final_helpers import get_vocab
from final import calc_CS
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
def hash_function(key, num_bins):
    # Hash function to map term and genre pair to a bin index
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % num_bins

def create_matrix(csv_file, num_bins):
    # Load the CSV containing n-grams and their Candidacy Scores
    parsed = generate_ngram_file(csv_file)
    df = pd.read_csv(parsed)
    CS_dict = calc_CS(csv_file)
    # Assuming `df` contains columns for each genre's Candidacy Score (CS) per term
    tokens = get_vocab(df)  # Assuming this gives you the list of unique tokens
    columns = ['id']
    
    df["Genres"] = df["Genres"].apply(
        lambda x: x.split(", ")
    )  # Assuming genres are comma-separated


    unique_genres = set(genre for genre_list in df['Genres'] for genre in genre_list)
    # Convert to a sorted list (optional)
    genres_arr = sorted(unique_genres)
    columns = columns + genres_arr 
    
    X = pd.DataFrame(columns=columns)
    Y = pd.DataFrame(columns=genres_arr)
    for index, row in df.iterrows():
        CS_mean = []
        Y_tags = []
        ngrams = row['Ngrams']
        genres = row['Genres']

        for genre in genres_arr:
            sum = 0
            if genre in genres:
                Y_tags.append(1)
            else:
                Y_tags.append(0)
            for token in ngrams:
                if token in CS_dict[genre].keys():
                    sum += CS_dict[genre][token]
            CS_mean.append(sum/len(ngrams))
        X.loc[len(X)] = [index] + CS_mean
        Y.loc[len(Y)] = Y_tags

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["Genres"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    ovr = OneVsRestClassifier(lr)

    # Train the classifier
    ovr.fit(X_train, y_train)
    y_pred = ovr.predict(X_test)

    from sklearn.metrics import classification_report

    # Print the classification report (Precision, Recall, F1-score for each genre)
    print(classification_report(y_test, y_pred, target_names=genres_arr))


            
        

        

    

# Example usage
create_matrix("eng_mov.csv", num_bins=14000)


 # Rows = documents, Columns = num_bins
