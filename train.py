import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\dadof\Downloads\reviews_data_dump\reviews_badminton\data.csv")

# Keep required columns
df = df[["Review text", "Ratings"]]

# Remove missing values
df.dropna(inplace=True)

# Convert rating to sentiment
# 4,5 = positive
# 1,2,3 = negative
df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["Review text"], df["sentiment"], test_size=0.2, random_state=42
)

# Convert text → numbers
tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Check accuracy
accuracy = model.score(tfidf.transform(X_test), y_test)
print("✅ Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("✅ model.pkl and tfidf.pkl saved successfully!")
