import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)

metadata_store = []

def add_vector(vector, metadata):
    vec = np.array([vector]).astype("float32")
    index.add(vec)
    metadata_store.append(metadata)

def search_vector(vector, k=3):
    vec = np.array([vector]).astype("float32")
    distances, indices = index.search(vec, k)

    results = []
    for i in indices[0]:
        if i < len(metadata_store):
            results.append(metadata_store[i])

    return results