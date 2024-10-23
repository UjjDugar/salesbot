import numpy as np
import pandas as pd
import re
import faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# this script creates the FAISS index that faiss_search can use to return the n most relevant products for a given query

# -----
INDEX = 'faiss_database.bin'

MAX_CHUNK_SIZE = 500 # experiment with this

NAME_COL = 'game_name'
DESC_COL = 'description'
PRICE_COL = 'price'
ID_COL = 'product_id'
SPECS_COL = None # important data to inclue in every chunk
# -----

def get_search_database(data): # assuming we have the columns in the format given above
    search_database = pd.DataFrame(columns=['id', 'entry'])
    for _, row in data.iterrows():
        master_id = row[ID_COL]
        description = row[DESC_COL].strip().replace('\n\n' ,'\n').replace('\n', ' ')
        description = recursive_split(description)

        for i, chunk in enumerate(description):
            if SPECS_COL:
                entry = {
                    'id': int(str(master_id) + f'{i+1:04d}'), # last 4 digits are chunk-specific (ignorable in overall search)
                    'entry': f"Product name: {row[NAME_COL]}. Price: {row[PRICE_COL]}\n{chunk}\n{row[SPECS_COL]}"
                }
            else:
                entry = {
                    'id': int(str(master_id) + f'{i+1:04d}'),
                    'entry': f"Product name: {row[NAME_COL]}. Price: {row[PRICE_COL]}\n{chunk}"
                }
        
            search_database.loc[len(search_database)] = entry

    return search_database

def save_faiss_index(database):
    ids = np.array([e['id'] for _, e in database.iterrows()]).astype(np.int64)
    entries = [e['entry'] for _, e in database.iterrows()]
    embeddings = np.array(model.encode(entries), dtype='float32')

    index = faiss.IndexFlatL2(embeddings.shape[1]) # can improve on this
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, INDEX)
    print(f'Successfully saved FAISS database to {INDEX}')

def save_faiss_index_ivf(database, nlist=10): # for IVF clustering (not implemented)
    ids = np.array([e['id'] for _, e in database.iterrows()]).astype(np.int64)
    entries = [e['entry'] for _, e in database.iterrows()]
    embeddings = np.array(model.encode(entries), dtype='float32')

    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embeddings.shape[1]), embeddings.shape[1], nlist)
    index.train(embeddings)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, INDEX)
    print(f'Successfully saved FAISS database to {INDEX}')


'''
query returns the ids
description gets chunked with the other cols as metadata in each chunk

!! the second faiss function uses IVF which speeds up for larger datasets

    - could use hierarchical chunking

    - find way to second-tier information such as reviews (or respond to input such as, 'what do people say about this?')

    - find way to register price correctly


can use weighted embeddings eg. to give metadata more importance
'''

def recursive_split(text, max_chunk_size=MAX_CHUNK_SIZE):
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = split_by_sentence(text, max_chunk_size)

    return chunks

def split_by_sentence(text, max_chunk_size):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    chunks = reduce_chunks(chunks, max_chunk_size)
    return chunks

def reduce_chunks(chunks, max_chunk_size):
    new_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            chunk1, chunk2 = split_in_half(chunk)
            new_chunks.extend(reduce_chunks([chunk1, chunk2], max_chunk_size))
        else:
            new_chunks.append(chunk)
    return new_chunks

def split_in_half(chunk):
    idx = len(chunk) // 2
    split_idx = chunk.rfind(' ', 0, idx)
    if split_idx == -1:
        split_idx = idx
    return chunk[:split_idx].strip(), chunk[split_idx:].strip()

# -----

data = pd.read_csv('product_data.csv')
database = get_search_database(data)
database.to_csv('search_database.csv')
save_faiss_index(database)