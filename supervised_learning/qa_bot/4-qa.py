#!/usr/bin/env python3
import os
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util

# Load BERT QA model
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

# Load semantic search model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

EXIT_WORDS = {"exit", "quit", "goodbye", "bye"}

def question_answer_from_text(question: str, reference: str):
    """Answer a question from a reference text using BERT QA."""
    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf"
    )
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]

    outputs = qa_model([input_ids, input_mask])
    start_logits, end_logits = outputs

    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    if start_index >= end_index:
        return None

    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens).strip()
    return answer if answer else None

def semantic_search(corpus_path: str, sentence: str) -> str:
    """Return the most semantically similar document to the sentence."""
    documents = []

    for fname in os.listdir(corpus_path):
        fpath = os.path.join(corpus_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append(text)

    if not documents:
        return None

    doc_embeddings = semantic_model.encode(documents, convert_to_tensor=True)
    query_embedding = semantic_model.encode(sentence, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    best_idx = int(similarities.argmax())
    return documents[best_idx]

def question_answer(corpus_path: str):
    """Interactive QA loop over multiple reference documents."""
    while True:
        question = input("Q: ").strip()
        if question.lower() in EXIT_WORDS:
            print("A: Goodbye")
            break

        # Find most relevant document
        reference = semantic_search(corpus_path, question)
        if not reference:
            print("A: Sorry, I do not understand your question.")
            continue

        # Get answer from BERT QA
        answer = question_answer_from_text(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
