import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# embeds the csv file as a FAISS search database and rewrites the csv file with an added product_id entry

DATA = sys.argv[1]
INDEX = f'{Path(DATA).stem}.bin'

data = pd.read_csv(DATA)
columns = list(data.columns)
model = SentenceTransformer('all-MiniLM-L6-v2')

def compress_row(row):
    chunks = []
    for c in columns:
        if pd.notnull(row[c]):
            chunks.append(f"{c.upper()}: {row[c]}")
    return " ".join(chunks)

product_info = data.apply(compress_row, axis=1)

def save_index():
    embeddings = model.encode(product_info.tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings, dtype='float32')
    product_ids = data['product_id'].values.astype(np.int64)

    index = faiss.IndexFlatL2(embeddings.shape[1]) # use better encoding here
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, product_ids)
    faiss.write_index(index, INDEX)
    print(f'Successfully saved FAISS database to {INDEX}')

if __name__ == '__main__':
    save_index()