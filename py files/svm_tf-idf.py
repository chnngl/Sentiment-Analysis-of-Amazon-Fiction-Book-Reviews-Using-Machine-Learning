import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import nltk
nltk.data.path.append("/usr/share/nltk_data")

# Data loading, cleaning, and extraction
br = pd.read_csv('/kaggle/input/amazon-books-reviews/Books_rating.csv')
br.head()

bd = pd.read_csv('/kaggle/input/amazon-books-reviews/books_data.csv')
bd.head()

books = pd.merge(br,bd, on = 'Title')
books.shape

df = books[['Title','review/score','review/text','categories']]
df.head()

df.drop_duplicates(inplace = True)
df.shape

df.dropna(inplace=True)
df.isna().sum()

df.shape

df = df[df['categories'].str.contains('Fiction', case=False, na=False)]
df.shape

df = df[~df['categories'].str.contains('Nonfiction', case=False, na=False)]
df.shape

df['categories'] = df['categories'].str.extract(r'\'(.*)\'')
df.head()

df['word_count'] = df['review/text'].apply(lambda x: len(x.split(' ')))
df.head()

# Negative class: 1 star
neg_df = df[df['review/score'].isin([1, 2])]

# Neutral class: 3 stars
neu_df = df[df['review/score'] == 3]

# Positive class: 4 or 5 stars
pos_df = df[df['review/score'].isin([4, 5])]

print(len(neg_df), len(neu_df), len(pos_df))

# Balanced three classes
neg_10k = neg_df.sample(n=10000, random_state=42)
neu_10k = neu_df.sample(n=10000, random_state=42)
pos_10k = pos_df.sample(n=10000, random_state=42)

balanced_df = pd.concat([neg_10k, neu_10k, pos_10k])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

def map_label(score):
    if score in [1, 2]:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

balanced_df['label'] = balanced_df['review/score'].apply(map_label)

balanced_df.head()

# Save data for other model training
balanced_df.to_csv('balanced30k.csv', index=False)

# Preprocessing
import nltk

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('wordnet')

print("All NLTK data downloaded!")

import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text_ml(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)                      # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)    # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)                    # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)                    # Remove punctuation except underscore
    text = re.sub(r'\d+', '', text)                        # Remove digits/numbers
    text = re.sub(r'\s+', ' ', text).strip()               # Normalize whitespace
     # Lemmatize
    words = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in words])
    return text

balanced_df['cleaned_text_ml'] = balanced_df['review/text'].apply(clean_text_ml)
balanced_df.head()

balanced_df.to_csv('balanced30k_SVM_cleaned.csv', index=False)

# Data split 7:2:1
from sklearn.model_selection import train_test_split

X = balanced_df['cleaned_text_ml']
y = balanced_df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)


print(f"Train size: {len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"Val size:   {len(X_val)} ({len(X_val)/len(X):.1%})")
print(f"Test size:  {len(X_test)} ({len(X_test)/len(X):.1%})")

# Find best config
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Define grid manually
param_grid = [
    #{'C': 0.01, 'stop_words': 'english', 'ngram_range': (1, 2), 'max_features': 20000},
    #{'C': 0.005, 'stop_words': 'english', 'ngram_range': (1, 2), 'max_features': 20000},
    #{'C': 0.005, 'stop_words': None, 'ngram_range': (1, 2), 'max_features': 15000},
    {'C': 0.01,   'stop_words': None, 'ngram_range': (1, 2), 'max_features': 20000},
    #{'C': 0.005,   'stop_words': None, 'ngram_range': (1, 2), 'max_features': 20000}
]

best_f1 = 0
best_model = None
best_config = None

for params in param_grid:
    print(f"Testing config: {params}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=params['ngram_range'],
            stop_words=params['stop_words']
        )),
        ('svm', LinearSVC(C=params['C']))
    ])
    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    
    report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    print(f"Macro F1: {macro_f1:.4f}")
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = pipeline
        best_config = params

print("\n Best config found:")
print(best_config)

# Use best config
y_train_pred = best_model.predict(X_train)

# Print classification report
print("Training Performance:")
print(classification_report(y_train, y_train_pred, zero_division=0))

# Predict on validation set using the best pipeline
y_val_pred = best_model.predict(X_val)

# Print classification report
print("Validation Performance:")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Predict on test data
y_test_pred = best_model.predict(X_test)

# Evaluate on test data
print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class_names = ['Negative', 'Neutral', 'Positive'] 

# Normalized by true class (each row sums to 1)
cm_norm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2], normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)


disp_norm.plot(values_format='.2f', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.show()