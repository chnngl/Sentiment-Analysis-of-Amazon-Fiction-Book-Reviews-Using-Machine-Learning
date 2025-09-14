#Kaggle Notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/dataset/balanced30k.csv')

df.head()

import re
def clean_for_transformer(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned_text'] = df['review/text'].apply(clean_for_transformer)
df.head()

from sklearn.model_selection import train_test_split
X = df['cleaned_text']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

from datasets import Dataset
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df   = pd.DataFrame({'text': X_val,   'label': y_val})
test_df  = pd.DataFrame({'text': X_test,  'label': y_test})

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset   = Dataset.from_pandas(val_df,   preserve_index=False)
test_dataset  = Dataset.from_pandas(test_df,  preserve_index=False)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

def tokenize(batch):
    return tokenizer( batch['text'],padding='max_length',truncation=True,max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
val_dataset   = val_dataset.map(tokenize,   batched=True, remove_columns=['text'])
test_dataset  = test_dataset.map(tokenize,  batched=True, remove_columns=['text'])

# Set PyTorch format
cols = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=cols)
val_dataset.set_format(type='torch',   columns=cols)
test_dataset.set_format(type='torch',  columns=cols)

model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy':  accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall':    recall_score(labels, preds, average='macro', zero_division=0),
        'f1':        f1_score(labels, preds, average='macro', zero_division=0),
    }

#Training arguments
training_args = TrainingArguments(
    output_dir="./roberta_fiction_sentiment/3",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Validation set
pred = trainer.predict(val_dataset)
logits = pred.predictions
y_true = pred.label_ids
y_pred = logits.argmax(axis=1)

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'],zero_division=0))


from transformers.trainer_utils import get_last_checkpoint
import os
import json
import matplotlib.pyplot as plt

log_file = "./roberta_fiction_sentiment/3/checkpoint-1971/trainer_state.json"

with open(log_file) as f:
    data = json.load(f)

steps = []
losses = []

for log in data["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

# Plot
plt.figure(figsize=(8, 5))
plt.plot(steps, losses, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.grid(True)
plt.legend()
plt.savefig("roberta_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# Evaluate on test set
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
preds = trainer.predict(test_dataset)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
plt.title("RoBERTa (test set) — Confusion Matrix")
plt.savefig("roberta_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Extract non-fiction book 
import pandas as pd
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

df = df[~df['categories'].str.contains('Fiction', case=False, na=False)]
df.shape

df['categories'] = df['categories'].str.extract(r'\'(.*)\'')
df.head()

# Negative class: 1 star
neg_df = df[df['review/score'].isin([1, 2])]

# Neutral class: 3 stars
neu_df = df[df['review/score'] == 3]

# Positive class: 4 or 5 stars
pos_df = df[df['review/score'].isin([4, 5])]

print(len(neg_df), len(neu_df), len(pos_df))

neg_1k = neg_df.sample(n=1000, random_state=42)
neu_1k = neu_df.sample(n=1000, random_state=42)
pos_1k = pos_df.sample(n=1000, random_state=42)

balanced_df = pd.concat([neg_1k, neu_1k, pos_1k])
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

balanced_df['cleaned_text'] = balanced_df['review/text'].apply(clean_for_transformer)
balanced_df.head()

balanced_df.shape

review_length = balanced_df['cleaned_text'].apply(lambda x: len(str(x).split()))
balanced_df.head()
review_length.describe()

balanced_df.to_csv('balanced3k_nonfiction.csv', index=False)

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Convert to Dataset
nf_df = balanced_df[['cleaned_text', 'label']].rename(columns={'cleaned_text':'text'})
nf_ds = Dataset.from_pandas(nf_df, preserve_index=False)

# Tokenize
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

nf_ds = nf_ds.map(tokenize, batched=True, remove_columns=['text'])

# Predict with trainer
preds = trainer.predict(nf_ds)
y_true = np.array(nf_ds['label'])
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(y_true, y_pred, zero_division=0,target_names=['Negative','Neutral','Positive']
))

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative','Neutral','Positive'])
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
plt.title('RoBERTa — Confusion Matrix (Non-Fiction 3k)')
plt.tight_layout()
plt.savefig('roberta_confmat_nonfiction3k.png', dpi=300)
plt.show()

# Goodreads db file

REVIEWS_DB = "/kaggle/input/books-dataset-goodreadsmay-2024/book_reviews.db"

import sqlite3
# Connect and inspect tables
con = sqlite3.connect(REVIEWS_DB)
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", con)
print(tables)

# See columns in the table
pd.read_sql_query("PRAGMA table_info(book_reviews);", con)

sql = """
SELECT review_content, review_rating
FROM book_reviews
WHERE review_content IS NOT NULL AND TRIM(review_content) <> ''
  AND review_rating IS NOT NULL
"""
df = pd.read_sql_query(sql, con)

# Extract the first digit from strings like "Rating 4 out of 5"
df['stars'] = df['review_rating'].str.extract(r'(\d)').astype(int)

# Map to sentiment labels
def map_rating(s):
    if s in [1, 2]: return 0   # Negative
    if s == 3:      return 1   # Neutral
    if s in [4, 5]: return 2   # Positive
df['label'] = df['stars'].apply(map_rating)

# Make a balanced 3k sample
neg = df[df['label']==0].sample(n=1000, random_state=42)
neu = df[df['label']==1].sample(n=1000, random_state=42)
pos = df[df['label']==2].sample(n=1000, random_state=42)

goodreads_balanced = pd.concat([neg, neu, pos]).sample(frac=1, random_state=42).reset_index(drop=True)
goodreads_balanced = goodreads_balanced.rename(columns={'review_content':'cleaned_text'})
print(goodreads_balanced.shape)

goodreads_balanced.head()

review_lengths = goodreads_balanced['cleaned_text'].apply(lambda x: len(str(x).split()))
print(review_lengths.describe())

from datasets import Dataset

goodreads_ds = Dataset.from_pandas(
    goodreads_balanced[['cleaned_text', 'label']].rename(columns={'cleaned_text':'text'}),
    preserve_index=False
)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

goodreads_tok = goodreads_ds.map(tokenize, batched=True, remove_columns=['text'])


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

preds = trainer.predict(goodreads_tok)
y_true = np.array(goodreads_tok['label'])
y_pred = np.argmax(preds.predictions, axis=1)

# Report
print(classification_report(
    y_true, y_pred, target_names=['Negative','Neutral','Positive'], digits=2, zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative','Neutral','Positive'])
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
plt.title('RoBERTa — Confusion Matrix (Goodreads 3k)')
plt.tight_layout()
plt.savefig('roberta_confmat_goodreads3k.png', dpi=300)
plt.show()

