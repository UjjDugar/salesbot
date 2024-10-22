import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

def compress_row(row):
    # Function that takes all the data and converts to a long string
    chunks = []
    for c in columns:
        if pd.notnull(row[c]):
            chunks.append(f"{c.upper()}: {row[c]}")
    return " ".join(chunks)


# Embeds the csv file as a FAISS search database and rewrites the csv file with an added product_id entry

# First argument should be a csv file
DATA = sys.argv[1]
INDEX = f'{Path(DATA).stem}.bin' 

data = pd.read_csv(DATA)
columns = list(data.columns)
model = SentenceTransformer('all-MiniLM-L6-v2')

# product_info is a list of strings, each pertaining to one item
product_info = data.apply(compress_row, axis=1) 

def save_index():
    embeddings = model.encode(product_info.tolist(), show_progress_bar=True) # Encodes each data item. Takes some time
    embeddings = np.array(embeddings, dtype='float32') # Converts it into numpy array for faiss
    
    product_ids = data['product_id'].values.astype(np.int64) # List of product IDs, to connect embeddings to IDs

    index = faiss.IndexFlatL2(embeddings.shape[1]) # Create a faiss index object using Euclidean distance 
    index = faiss.IndexIDMap(index) # Allows you to add IDs
    index.add_with_ids(embeddings, product_ids) # Add embeddings and corresponding IDs to the index 
    faiss.write_index(index, INDEX) # Write the index to a file
    print(f'Successfully saved FAISS database to {INDEX}')

if __name__ == '__main__':
    # print(product_info)
    save_index()