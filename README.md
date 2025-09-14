# Sentiment Analysis of Amazon Fiction Book Reviews Using Machine Learning

This repository contains the implementation and experiments for my MSc dissertation project on **Sentiment Analysis of Amazon Fiction Book Reviews Using Machine Learning**.  
The study compares **traditional machine learning, deep learning, and transformer-based models** on fiction book reviews, with additional evaluation of the best model RoBERTa on non-fiction and Goodreads reviews.

---

## Data Sources

The datasets used in this study were obtained from Kaggle:

- [Amazon Book Reviews (Fiction & Non-Fiction)](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data)  
- [Goodreads Book Reviews](https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024)


Data preparation:

- A balanced dataset of 30,000 Amazon fiction reviews was obtained (10,000 per class: positive, neutral, negative).
- An additional 3,000 Amazon non-fiction reviews and 3,000 Goodreads reviews were used for robustness testing (1000 per class).

Two processed files are reused during training:
- balanced30k.csv → Extracted balanced dataset containing 30k fiction book reviews.
- balanced30k_processed_dl.csv → Preprocessed version used by deep learning models.

## Models Implemented

### 1. Traditional Machine Learning
- Support Vector Machine (SVM) with TF–IDF features
- Hyperparameters tuned: C values (0.005–1.0), n-gram ranges (1–3), max features (10k–30k)

### 2. Deep Learning
- **CNN**, **LSTM**, **BiLSTM**, and hybrid **CNN-(Bi)LSTM** models  
- Pretrained **100-dimensional GloVe embeddings** (trainable)  
- Training optimizers: Adam with EarlyStopping and ReduceLROnPlateau
- Hyperparameters tuned: kernel sizes (3, 5, 7), recurrent units (64/128), dropout (0.3–0.6), epochs (30–50), etc  

### 3. Transformer Models
- **BERT (base-ncased & multilingual)**  
- **RoBERTa (best performing, 77% accuracy on fiction dataset)**  
- Fine-tuning with Hugging Face Transformers
- Hyperparameters tuned: learning rate

 ## Key Findings
- **RoBERTa** achieved the highest overall accuracy (**77%**) on Amazon fiction reviews.  
- **BiLSTM** was the strongest among deep learning models (~69%)
- Hybrid models CNN-(Bi)LSTM didn't perform better than single variant.  
- Neutral class is the most difficult to classify.  
- RoBERTa's Performance dropped on longer Goodreads reviews, highlighting challenges with text truncation in transformers.  

## Repository Structure
.
```
├── data/                          
│   ├── balanced30k.csv            # balanced 30k reviews 
│   └── balanced30k_processed_dl.csv  # preprocessed for DL
├── src/                           # runnable Python scripts
│   ├── bert-base-multilingual-uncased_finetune.py
│   ├── bert-base-uncased_finetune.py
│   ├── bilstm_glove.py
│   ├── cnn_glove.py
│   ├── cnn-bilstm_glove.py
│   ├── cnn-lstm_glove.py
│   ├── lstm_glove.py
│   ├── roberta-base_finetune.py
│   ├── svm-tf-idf.py
│   ├── zero_shot_test_transformer.py
├── notebooks/                     # GoogleColab and Kaggle notebooks
│   ├── bert-base-multilingual-uncased_finetune.ipynb
│   ├── bert-base-uncased_finetune.ipynb
│   ├── bilstm_glove.ipynb
│   ├── cnn_glove.ipynb
│   ├── cnn-bilstm_glove.ipynb
│   ├── cnn-lstm_glove.ipynb
│   ├── lstm_glove.ipynb
│   ├── roberta-base_finetune.ipynb
│   ├── svm-tf-idf.ipynb
│   ├── zero_shot_test_transformer.ipynb
└── README.md
```

## How to Run
1. Clone this repo:
   git clone https://github.com/yourusername/fiction-sentiment-analysis.git
2. Option A: Run on Kaggle / Google Colab (Recommended)
Open the notebooks (.ipynb) in Kaggle or Google Colab.
Upload the dataset files (balanced30k.csv, balanced30k_processed_dl.csv) to the notebook environment.
Update file paths if needed (e.g., /kaggle/input/...).
For deep learning or transformer models, turn on the GPU setting to accelerate the training
Run cells directly to reproduce results.
3. Option B: Run locally (Python scripts)
Use the .py files under src folder and update data file paths.
