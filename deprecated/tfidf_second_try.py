from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
pd.options.display.max_rows = 10
corpus_name = sys.argv[1]

dir_path  = './' + corpus_name

text_files = glob.glob(f"{dir_path}/*.txt")
text_titles = [Path(text).stem for text in text_files]


# Step 1: Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(input="filename", stop_words='english', max_features=1000)

# Step 2: Fit the vectorizer to the movie synopses and transform the text data
X_tfidf = vectorizer.fit_transform(text_files)

# Step 3: Convert the result to a dense matrix (for easier inspection)
X_tfidf_dense = X_tfidf.toarray()

# Step 4: Show the shape of the matrix (documents x features)
print("Shape of TF-IDF matrix: ", X_tfidf_dense.shape)

tfidf_df = pd.DataFrame(X_tfidf.toarray(), index=text_titles, columns=vectorizer.get_feature_names_out())

tfidf_df.to_csv('wawqa.csv')



corpus_df = pd.read_csv('english_movies.csv')
genres = corpus_df['genres'].to_numpy()

genre_tags = []

for genre in genres:
    try:
        genre_tags.append(genre.split(', '))
    except:
        
        pass

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(genre_tags)  # Convert to binary matrix


# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Step 2: Create an SVM model with One-vs-Rest strategy
svm = SVC(kernel='linear', C=1000)  # You can change kernel and C based on your data
ovr_classifier = OneVsRestClassifier(svm)

# Step 3: Train the model
ovr_classifier.fit(X_train, y_train)

# Step 4: Predict on the test data
y_pred = ovr_classifier.predict(X_test)

print(y_pred)

# Step 5: Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))  # Overall accuracy for multi-label
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

# Step 6: Making predictions for new data (example)
new_text = ["A thrilling action-packed sci-fi adventure."]
new_tfidf = np.random.rand(1, 1000)  # Replace with actual TF-IDF vectorization of new text

# Predict the genre(s) for the new movie
new_pred = ovr_classifier.predict(new_tfidf)
predicted_labels = mlb.inverse_transform(new_pred)  # Convert binary labels back to genre names

print("Predicted Genres for New Movie: ", predicted_labels)