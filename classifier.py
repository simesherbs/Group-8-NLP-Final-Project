from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

corpus = pd.read_csv('english_movies_clean.csv')
genre_terms = pd.read_csv('CT_withngrams.csv')
corpus["genres"] = corpus["genres"].apply(
    lambda x: x.split(", ")
)

vocabulary = []
def infer_tags(q):
    q_vec = vectorizer.transform([q])
    q_pred = classifier.predict(q_vec)

    return mlb.inverse_transform(q_pred)

for id, vocab in genre_terms.iterrows():
    temp = vocab['terms'].split(', ')
    vocabulary += temp[0:len(temp)-1]
vocabulary = set(vocabulary)
vectorizer = CountVectorizer(vocabulary=vocabulary, binary=False)
X = vectorizer.fit_transform(corpus['overview'])
# Convert multi-label genres to binary arrays
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(corpus['genres'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Use Logistic Regression with a OneVsRest strategy for multi-label classification
classifier = OneVsRestClassifier(SVC())
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Print classification report
res = (classification_report(y_test, y_pred, target_names=mlb.classes_))


resdf = pd.DataFrame(res).transpose()

resdf.to_csv('results.csv')

for i in range(5): 
  k = X_test.sample(1).index[0] 
  print("Movie: ", corpus['id'][k], "\nPredicted genre: ", infer_tags(X_test[k])), print("Actual genre: ",corpus['genres'][k], "\n")