import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer   # or CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\HP\Desktop\spam.csv")

# Dataset columns: text, spam (0 = ham, 1 = spam)
df = df[['text', 'spam']]
df.rename(columns={'text': 'message', 'spam': 'label'}, inplace=True)

# -----------------------------
# Handle missing values
# -----------------------------
df.dropna(subset=['message', 'label'], inplace=True)
df['message'] = df['message'].astype(str)

# -----------------------------
# Remove duplicates
# -----------------------------
df.drop_duplicates(inplace=True)

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()                       # lowercase
    text = re.sub(r'subject:', '', text)      # remove email subject word
    text = re.sub(r'[^a-zA-Z]', ' ', text)    # remove punctuation & special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)

# -----------------------------
# Train-test split
# -----------------------------
X = df['clean_message']
y = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Model training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_vec)

# -----------------------------
# Evaluation metrics
# -----------------------------
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Logistic Regression Model
# -----------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_vec, y_train)

y_pred_lr = log_reg.predict(X_test_vec)

print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1 Score :", f1_score(y_test, y_pred_lr))

print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, y_pred_lr))

# -----------------------------
# Support Vector Machine (SVM)
# -----------------------------
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

y_pred_svm = svm_model.predict(X_test_vec)

print("\n--- SVM Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall   :", recall_score(y_test, y_pred_svm))
print("F1 Score :", f1_score(y_test, y_pred_svm))

print("\nClassification Report (SVM):\n")
print(classification_report(y_test, y_pred_svm))

# -----------------------------
# F1-score comparison
# -----------------------------
f1_nb = f1_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred_lr)
f1_svm = f1_score(y_test, y_pred_svm)

print("\n--- F1 Score Comparison ---")
print("Naive Bayes F1 Score       :", f1_nb)
print("Logistic Regression F1    :", f1_lr)
print("SVM F1 Score              :", f1_svm)

# Determine best model
f1_scores = {
    "Naive Bayes": f1_nb,
    "Logistic Regression": f1_lr,
    "SVM": f1_svm
}

best_model = max(f1_scores, key=f1_scores.get)

print("\nMost accurate model based on F1-score:")
print(f"{best_model} gives the highest F1-score ({f1_scores[best_model]:.4f})")


# -----------------------------
# Custom prediction function
# -----------------------------
def predict_email(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example
test_email = "Congratulations! You have won a free prize. Click now!"
print("\nTest Email Prediction:", predict_email(test_email))
