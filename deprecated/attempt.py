from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import sys
from sklearn.multiclass import OneVsRestClassifier

corpus_name = sys.argv[1]
# Step 1: Extract Top Terms (Assumes `importance_df` from the previous code)
top_n_terms = ['love', 'murder', 'war', 'life', 'magic', 'world', 'find', 'stori', 'documentari', 'adventur']

movies = pd.read_csv(corpus_name + '_clean.csv')
# Step 2: Create a Reduced TF-IDF Matrix
# Filter terms based on the top terms identified
vectorizer = TfidfVectorizer(vocabulary=top_n_terms)
X_reduced_tfidf = vectorizer.fit_transform(movies['overview'])

genres = []
for i in movies["genres"]:
    genres.append(str(i).split(", "))

multilabel_binarizer = MultiLabelBinarizer()
movies['cleaned_genres'] = genres
Y_genres = multilabel_binarizer.fit_transform(movies['cleaned_genres'])

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced_tfidf, Y_genres, test_size=0.3, random_state=42)

# Step 4: Train Final Classifier (SVM in this case)
final_classifier = Pipeline([
    ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42)))
])

final_classifier.fit(X_train, y_train)

# Step 5: Evaluate Performance
y_pred = final_classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=multilabel_binarizer.classes_))
