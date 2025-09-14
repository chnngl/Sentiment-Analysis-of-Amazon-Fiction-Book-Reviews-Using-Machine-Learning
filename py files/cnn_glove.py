from google.colab import files
import pandas as pd
uploaded = files.upload()

import pandas as pd

# Load the CSV file directly by filename
df = pd.read_csv('balanced30k.csv')

# Check the first few rows to confirm it loaded correctly
print(df.head())

# Preprocessing

!pip install contractions

import re
import numpy as np
import pandas as pd
import contractions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text_dl(text):
    text = str(text).lower()
    text = contractions.fix(text) # Expand contractions
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r"([!?.,'])", r" \1 ", text)  # Space out punctuation
    text = re.sub(r"[^a-zA-Z0-9!?.,' ]", '', text)  # Remove non-ASCII chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text_dl'] = df['review/text'].apply(clean_text_dl)

df.head()

df.to_csv('balanced30k_processed_dl.csv', index=False)

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset
X = df['cleaned_text_dl']
y = df['label']

# Split into train (70%), val (15%), test (15%) with stratification to keep class balance
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

print(f'Train size: {len(X_train)}')
print(f'Validation size: {len(X_val)}')
print(f'Test size: {len(X_test)}')

review_lengths = df['cleaned_text_dl'].apply(lambda x: len(x.split()))
review_lengths.describe()

# Adjust based on dataset
max_vocab_size = 20000
max_sequence_length = 250

# Tokenizing and padding
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

print(f'Example tokenized sequence: {X_train_seq[0]}')
print(f'Example padded sequence shape: {X_train_pad.shape}')

# Utilizing GloVe
!wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

print(f"Loaded {len(embedding_index)} word vectors.")

embedding_dim = 100
word_index = tokenizer.word_index
num_words = min(max_vocab_size, len(word_index) + 1)

# Build embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print(f'Embedding matrix shape: {embedding_matrix.shape}')

# Model training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

model_cnn = Sequential()

model_cnn.add(Embedding(input_dim=num_words,
                        output_dim=embedding_dim,
                        input_length=max_sequence_length,
                        trainable=True))  # learn embeddings from scratch

model_cnn.add(Dropout(0.6))

model_cnn.add(Conv1D(128, kernel_size=5, activation='relu', kernel_regularizer=l2(1e-4)))

model_cnn.add(MaxPooling1D(pool_size=2))

model_cnn.add(GlobalMaxPooling1D())

model_cnn.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))

model_cnn.add(Dropout(0.6))

model_cnn.add(Dense(3, activation='softmax', kernel_regularizer=l2(1e-4)))

optimizer = Adam(learning_rate=3e-5, clipnorm=1.0)
model_cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()

# Optimization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss',
                           patience=3,    # stop if no improvement after 4 epochs
                           restore_best_weights=True,  # keep best model weights
                           verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=1e-6,
                              verbose=1)
history = model_cnn.fit(
    X_train_pad,
    y_train,
    epochs=50,          # adjusted from 10 to 50
    batch_size=64,      # adjusted from 64
    validation_data=(X_val_pad, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(7, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("model_accuracy_cnn.png", dpi=300, bbox_inches='tight') 
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(7, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("model_loss_cnn.png", dpi=300, bbox_inches='tight')      
plt.show()

import numpy as np

# Get predicted probabilities
y_val_probs = model_cnn.predict(X_val_pad)

# Convert to predicted class labels (0, 1, 2)
y_val_pred = np.argmax(y_val_probs, axis=1)

# Convert to numpy arrays
y_val_true = np.array(y_val)

# Find misclassified indices
wrong_indices = np.where(y_val_pred != y_val_true)[0]
print(f"Number of misclassified samples: {len(wrong_indices)}")

for idx in wrong_indices[:10]:  # find 10 misclassification
    print(f"\n Review: {X_val.iloc[idx]}")
    print(f" True label: {y_val_true[idx]}")
    print(f" Predicted:  {y_val_pred[idx]}")

# Get predicted class indices from softmax probabilities
y_val_probs = model_cnn.predict(X_val_pad)
y_val_pred = np.argmax(y_val_probs, axis=1)

# Evaluate against true labels
print(classification_report(y_val, y_val_pred, zero_division=0))


