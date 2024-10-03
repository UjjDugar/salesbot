import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sys

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

RETURN_K = 3

# get the model to rewrite the query, or make the query using accumulated context about the product
# modify funct below to return top 3 ids

def faiss_search(query, k=10):
    query_vector = embedding_model.encode([query])
    index = faiss.read_index('product_data.bin')
    d, i = index.search(query_vector, k=k)
    i, d = np.array(i).flatten(), np.array(d).flatten()
    ids = set()
    return_ids = []

    for idx in i:
        id = idx//10000 # remove trailing 4 identifiers
        if id not in ids:
            ids.add(id)
            return_ids.append(id)

        if len(return_ids) == RETURN_K:
            break

    if len(return_ids) < RETURN_K:
        return faiss_search(query, k=k*2)
    
    return return_ids


if __name__ == '__main__':
    query = 'Looking for an action shooter game under 20 pounds.'
    query = 'third-person action shooter game'
    index = faiss.read_index('product_data.bin')
    product_ids = faiss_search(query)
    print(f"Product IDs for query '{query}': {product_ids}")