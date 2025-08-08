# memory.py

import faiss
import numpy as np
import os
import pickle

EMBED_DIM = 384
INDEX_FILE = "memory.index"
TEXT_FILE = "memory_texts.pkl"

def create_index():
    return faiss.IndexFlatL2(EMBED_DIM)

def save_index(index, texts):
    faiss.write_index(index, INDEX_FILE)
    with open(TEXT_FILE, "wb") as f:
        pickle.dump(texts, f)

def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(TEXT_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(TEXT_FILE, "rb") as f:
            texts = pickle.load(f)
        return index, texts
    return create_index(), []
