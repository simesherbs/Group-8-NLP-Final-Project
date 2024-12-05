from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train a Random Forest Classifier
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y_genres, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with terms and their importance scores
feature_importance_df = pd.DataFrame({
    'Term': terms,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the most important features
print(feature_importance_df_sorted.head(10))
