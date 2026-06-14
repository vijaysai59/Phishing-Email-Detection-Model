import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Custom URL Feature Extractor
# -----------------------------
class URLFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        feature_list = []

        suspicious_words = [
            "verify",
            "login",
            "password",
            "account",
            "bank",
            "urgent",
            "click",
            "update",
            "confirm",
            "suspended",
            "security"
        ]

        for email in X:

            urls = re.findall(r'https?://\S+|www\.\S+', email)

            url_count = len(urls)

            keyword_count = 0

            for word in suspicious_words:
                keyword_count += email.lower().count(word)

            email_length = len(email)

            feature_list.append([
                url_count,
                keyword_count,
                email_length
            ])

        return np.array(feature_list)


# -----------------------------
# Load Dataset
# -----------------------------
print("Loading Dataset...")

data = pd.read_csv("dataset/emails.csv")

print(data.head())

X = data["email"]

y = data["label"]


# -----------------------------
# TF-IDF
# -----------------------------
print("\nExtracting Text Features...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_text = tfidf.fit_transform(X)

# -----------------------------
# URL Features
# -----------------------------
feature_extractor = URLFeatures()

X_url = feature_extractor.transform(X)

# -----------------------------
# Combine Features
# -----------------------------
X_combined = hstack((X_text, X_url))

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
print("\nTraining Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed!")

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : {:.2f}%".format(accuracy * 100))

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing","Safe"],
    yticklabels=["Phishing","Safe"]
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# -----------------------------
# Test New Email
# -----------------------------
def predict_email(email):

    text = tfidf.transform([email])

    url = feature_extractor.transform([email])

    final = hstack((text, url))

    prediction = model.predict(final)[0]

    return prediction


while True:

    print("\n==============================")
    print("PHISHING EMAIL DETECTOR")
    print("==============================")

    email = input("\nEnter Email Text:\n")

    result = predict_email(email)

    if result == "phishing":
        print("\nPrediction : PHISHING EMAIL")
    else:
        print("\nPrediction : SAFE EMAIL")

    choice = input("\nTest Another Email? (y/n): ")

    if choice.lower() != "y":
        break

print("\nProgram Ended.")