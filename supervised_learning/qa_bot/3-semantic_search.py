#!/usr/bin/env python3
import os
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(corpus_path: str, sentence: str) -> str:
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): path to directory containing reference documents
        sentence (str): query sentence

    Returns:
        str: text of the most similar reference document
    """
    documents = []
    doc_names = []

    # Read all files in corpus_path
    for fname in os.listdir(corpus_path):
        fpath = os.path.join(corpus_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append(text)
                    doc_names.append(fname)

    if not documents:
        return None

    # Encode documents and query
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    # Find best match
    best_idx = int(similarities.argmax())
    return documents[best_idx]
