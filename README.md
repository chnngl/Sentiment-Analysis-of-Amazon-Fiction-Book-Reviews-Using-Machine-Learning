# Sentiment Analysis of Amazon Fiction Book Reviews Using Machine Learning

This repository contains the implementation and experiments for my MSc dissertation project on **Sentiment Analysis of Book Reviews**.  
The study compares **traditional machine learning, deep learning, and transformer-based models** on fiction book reviews, with additional evaluation of the best model on non-fiction and Goodreads reviews.

---

## Data Sources

The datasets used in this study were obtained from Kaggle:

- [Amazon Books Reviews (Fiction/Non-Fiction)](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data)  
- [Goodreads Books Dataset (May 2024)](https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024)

For this project:
- A balanced dataset of **30,000 Amazon fiction reviews** (10k per class: positive, neutral, negative) was obtained.  
- An additional **3,000 non-fiction Amazon reviews** and **3,000 Goodreads reviews** were used for robustness testing.

---

## Models Implemented

### 1. Traditional Machine Learning
- Support Vector Machine (SVM) with TF–IDF features  

### 2. Deep Learning
- **CNN**, **LSTM**, **BiLSTM**, and hybrid **CNN-(Bi)LSTM** models  
- Pretrained **100-dimensional GloVe embeddings** (trainable)  
- Training optimizers: Adam with EarlyStopping and ReduceLROnPlateau  

### 3. Transformer Models
- **BERT (base, uncased & multilingual)**  
- **RoBERTa** (best performing, 77% accuracy on fiction dataset)  
- Fine-tuning with Hugging Face Transformers

 ## Key Findings
- **RoBERTa** achieved the highest overall accuracy (**77%**) on Amazon fiction reviews.  
- **BiLSTM** was the strongest among deep learning models (~69%), and hybrid models, CNN-(Bi)LSTM didn't perform better than single variant.  
- Neutral class is the most difficult to classify.  
- RoBERTa's Performance dropped on longer Goodreads reviews, highlighting challenges with text truncation in transformers.  

---

## How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/fiction-sentiment-analysis.git

2. Run the code on Google Colab or Kaggle
