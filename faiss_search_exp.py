import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tkinter import *
from tkinter import scrolledtext

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
RETURN_K = 3

# FAISS search function
def faiss_search(query, k=10):
    query_vector = embedding_model.encode([query])
    index = faiss.read_index('product_data.bin')
    d, i = index.search(query_vector, k=k)
    i, d = np.array(i).flatten(), np.array(d).flatten()
    ids = set()
    return_ids = []

    for id_with_extra_numbers in i:
        id = id_with_extra_numbers // 10000  # Remove trailing 4 identifiers
        if id not in ids:
            ids.add(id)
            return_ids.append(id)

        if len(return_ids) == RETURN_K:
            break

    if len(return_ids) < RETURN_K:
        return faiss_search(query, k=k*2)
    
    return return_ids

# Function to display results in the GUI
def display_results():
    query = query_entry.get()
    product_ids = faiss_search(query)

    # Clear previous results
    result_text.delete(1.0, END)

    # Read the product data
    data = pd.read_csv('product_data.csv')

    # Display the product IDs and corresponding data
    result_text.insert(END, f"Product IDs for query '{query}': {product_ids}\n\n")

    for product_id in product_ids:
        unclean_row = data[data['product_id'] == product_id].to_string(index=False).strip()
        clean_row = "".join(unclean_row.split('  '))
        result_text.insert(END, clean_row + "\n\n")

if __name__ == '__main__':

    root = Tk()
    root.geometry("600x400")
    root.title("FAISS Search Experiment")
    # Query entry label
    query_label = Label(root, text="Enter your search query:")
    query_label.pack(pady=5)
    # Query entry field
    query_entry = Entry(root, width=50)
    query_entry.pack(pady=5)
    # Button to perform the search
    search_button = Button(root, text="Search", command=display_results)
    search_button.pack(pady=10)
    # Text area to display the results
    result_text = scrolledtext.ScrolledText(root, width=240, height=150)
    result_text.pack(pady=10)
    # Run the application
    root.mainloop()
