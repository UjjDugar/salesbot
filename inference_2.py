from fastapi import FastAPI, Request
#from fastapi.lifespan import Lifespan
from huggingface_hub import InferenceClient
import os
import pandas as pd
import numpy as np
from faiss_search import faiss_search
import chromadb
from usingllm import GPT
from rich import traceback 
traceback.install()

model = 'gpt-3.5-turbo' # gpt-3.5-turbo' or gpt-4o or test
prompt = None # user input
product_data = 'product_data.csv'

product_database = pd.read_csv(product_data)
chroma_client = chromadb.Client()
memory = chroma_client.create_collection(name='local-memory')
product_memory = []
chat_count = 0


def store_product(id): # IMPLEMENT THIS LATER
    entry = product_database.loc[product_database['product_id'] == id]
    name = entry['game_name'].item()
    price = entry['price'].item()
    link = entry['link'].item()
    description = entry['description'].item() # segment in chunks of 500 chars

    memory.add(
        documents=[price],
        ids=[f'product_{len(product_memory)}'],
        metadatas=[{"category": "products", "product_name": name}]
    )
    memory.add(
        documents=[link],
        ids=[f'product_{len(product_memory)+1}'],
        metadatas=[{"category": "products", "product_name": name}]
    )
    memory.add(
        documents=[description],
        ids=[f'product_{len(product_memory)+2}'],
        metadatas=[{"category": "products", "product_name": name}]
    )

def store_product(id):
    entry = product_database.loc[product_database['product_id'] == id]
    entry = ', '.join([f"'{col.upper()}': '{value}'" for col, value in entry.iloc[0].items() if col not in ['Unnamed: 0', 'product_id']])

    memory.add(
        documents=[entry],
        ids=[f'product_{len(product_memory)}'],
        metadatas=[{"category": "products"}]
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
    print(f'product ids: {product_ids}')
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
    )['documents']

    store_prompt(prompt)

    # prompt = f'Instructions: Answer as a helpful assistant in one or two short sentences. Use bullet points when relevant. Give a few suggestions for the product.\n\nUser input: {prompt}\n\nPrevious inputs: {history_results}\n\nBackground information: {product_results}'
    prompt = f'Instructions: Answer as a helpful friend giving a recommendation. Dont exceed 35 words. Only talk about the one most relevant product. No need to include any link or prices. \n\nUser input: {prompt}\n\nPrevious inputs: {history_results}\n\nBackground information: {product_results}'

    llm = GPT(model)
    response = llm.write_message(prompt)

    # chat_count += 1
    return response, 8, (1,2)


if __name__ == '__main__':
    print(generate('I am looking for an action shooting game.'))
