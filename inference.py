from fastapi import FastAPI, Request
#from fastapi.lifespan import Lifespan
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
from faiss_search import faiss_search
import chromadb
import textwrap
from usingllm import GPT
from rich import traceback 
traceback.install()

llm_model = 'gpt-3.5-turbo' # gpt-3.5-turbo' or gpt-4o or test
product_data = 'product_data.csv'
chunk_size = 500 # max character len for local memory

product_database = pd.read_csv(product_data)
chroma_client = chromadb.Client()
memory = chroma_client.create_collection(name='local-memory')
product_memory = []
chat_count = 0

model = SentenceTransformer('all-MiniLM-L6-v2')


def store_product(id):
    entry = product_database.loc[product_database['product_id'] == id]
    entry = ', '.join([f"'{col.upper()}': '{value}'" for col, value in entry.iloc[0].items() if col not in ['Unnamed: 0', 'product_id']])

    memory.add(
        documents=[entry],
        ids=[f'product_{len(product_memory)}'],
        metadatas=[{"category": "products", "product_id": id}]
    )


def store_prompt(text):
    memory.add(
        documents=[text],
        ids=[f'prompt_{chat_count}'],
        metadatas=[{"category": "user_inputs"}]
    )

# -- detect if prompt is relevant & if it requires a FAISS search
# -- preprocess the prompt

def generate(prompt):
    product_ids, _ = faiss_search(prompt) # also returns distances
    #print(f'FAISS product ids: {product_ids}')
    for id in product_ids.tolist():
        if id not in product_memory:
            store_product(id)
            product_memory.append(id)

    history_results = memory.query(
        query_texts=[prompt],
        n_results=3,
        where={"category": "user_inputs"}
    )['documents']

    product_results = memory.query(
        query_texts=[prompt],
        n_results=3,
        where={"category": "products"}
    )
    ids = []
    metadata = product_results['metadatas'][0]
    for dict in metadata:
        ids.append(dict['product_id'])

    #print(f'ChromaDB product ids: {ids}')
    product_results = product_results['documents'][0]

    store_prompt(prompt) # store the user input in the temporary memory

    prompt = f'Instructions: Instructions: Answer as a helpful friend giving a recommendation in casual language. Dont exceed 35 words. Only talk about the one most relevant product. If no product is very relevant, mention that. No need to include any link or prices.\n\nUser input: {prompt}\n\nPrevious inputs: {history_results}\n\nBackground information: {product_results}'

    llm = GPT(llm_model)
    response = llm.write_message(prompt)
    
    encoded_response = model.encode(response, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    sims = {}
    for id, document in zip(ids, product_results): # note ids and documents must not have been reordered for this to work
        document = model.encode(document, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        sim = cosine_similarity(encoded_response, document)
        sims[id] = sim
    suggestion_ids = sorted(sims, key=sims.get)[:2]
    relevant_id = max(sims, key=sims.get)

    # chat_count += 1
    return response, relevant_id, suggestion_ids


if __name__ == '__main__':
    print(generate('I am looking for an action shooting game.'))
