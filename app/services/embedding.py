from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    return model.encode(text).tolist()

def clean_text(text):
    return text.replace("  ", " ").strip().lower()