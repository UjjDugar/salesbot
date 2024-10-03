from fastapi import FastAPI, Request
#from fastapi.lifespan import Lifespan
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
import chromadb
import textwrap
from usingllm import GPT
from rich import traceback 
traceback.install()
from sentence_transformers import SentenceTransformer
from faiss_search import faiss_search

llm_model = 'gpt-3.5-turbo' # gpt-3.5-turbo' or gpt-4o or test
llm = GPT(llm_model)

product_data = 'product_data.csv'
chunk_size = 500 # max character len for local memory

product_database = pd.read_csv(product_data)
user_history = []
embedder_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_info(id):
    entry = product_database.loc[product_database['product_id'] == id]
    entry = ', '.join([f"'{col.upper()}': '{value}'" for col, value in entry.iloc[0].items() if col not in ['Unnamed: 0', 'product_id']])
    return entry


# -- detect if prompt is relevant & if it requires a FAISS search
# -- preprocess the prompt

def generate(prompt, previous_response=''):
    pre_input = f"Create a concise one-line prompt to feed into a FAISS search to get a relevant product. Strip all unnecessary words such as 'I am looking for...' and only keep useful keywords. Take into account all of the following information:\n\nUser query: {prompt}\n\Previous user queries: {' '.join(user_history)}\n\nPrevious response: {previous_response}"
    faiss_input = llm.write_message(pre_input)
    product_ids = faiss_search(faiss_input)
    user_history.append(prompt)

    product_results = [get_info(id) for id in product_ids]
    prompt = f'Instructions: Answer as a helpful assistant in one or two short sentences. Use bullet points when relevant. Only select the most relevant product from the list.\n\nUser input: {prompt}\n\nPrevious inputs: {' '.join(user_history)}\n\nBackground information: {'\n\n'.join(product_results)}'
    response = llm.write_message(prompt)
    
    encoded_response = embedder_model.encode(response, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    sims = {}
    for id, document in zip(product_ids, product_results):
        document = embedder_model.encode(document, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        sim = cosine_similarity(encoded_response, document)
        sims[id] = sim
    suggestion_ids = sorted(sims, key=sims.get)[:2]
    relevant_id = max(sims, key=sims.get)

    return response, relevant_id, suggestion_ids

if __name__ == '__main__':
    print(generate('I am looking for an action shooting game.'))
