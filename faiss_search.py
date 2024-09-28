import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import sys

nltk.download('punkt')
nltk.download('stopwords')

model = SentenceTransformer('all-MiniLM-L6-v2')

RETURN_K = 3

def preprocess_query(query): # REMOVED
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query.lower())
    filtered_query = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_query)

def faiss_search(query):
    query_vector = model.encode([query])
    index = faiss.read_index('product_data.bin')
    d, i = index.search(query_vector, k=RETURN_K)
    i = np.array(i).flatten() # should be product ids if database has encoded ids
    d = np.array(d).flatten()
    return i, d

if __name__ == '__main__':
    query = sys.argv[1]
    index = faiss.read_index('product_data.bin')
    product_ids, scores = faiss_search(query)
    print(f"Product IDs for query '{query}': {product_ids}")
    print(f"Similarity scores: {scores}")