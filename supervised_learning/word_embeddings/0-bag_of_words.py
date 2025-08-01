import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    # Tokenize all words
    words = []
    tokenized_sentences = []

    for sentence in sentences:
        # Tokenize and lowercase
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        # Optional: remove 's' tokens
        tokens = [t for t in tokens if t != 's']
        tokenized_sentences.append(tokens)
        words.extend(tokens)

    # Build vocabulary if not provided
    if vocab is None:
        features = sorted(set(words))
    else:
        features = vocab

    # Build embeddings
    embeddings = []
    for tokens in tokenized_sentences:
        vector = [tokens.count(word) for word in features]
        embeddings.append(vector)

    return np.array(embeddings), features
