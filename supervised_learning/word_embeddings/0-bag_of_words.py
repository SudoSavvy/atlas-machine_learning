import numpy as np
import re

def bag_of_words(sentences):
    vocab = set()
    tokenized_sentences = []
    
    for sentence in sentences:
        # Tokenize: keep only whole words, lowercase
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        tokenized_sentences.append(tokens)
        vocab.update(tokens)
    
    vocab = sorted(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    matrix = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_to_index:
                matrix[i][word_to_index[token]] += 1
    
    return matrix, vocab
