#!/usr/bin/env python3
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer

# Load tokenizer and model once
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

EXIT_WORDS = {"exit", "quit", "goodbye", "bye"}

def question_answer(question: str, reference: str):
    """Answer a question from a reference text using BERT QA."""
    # Encode
    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf"
    )
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]

    # Run model
    outputs = model([input_ids, input_mask])
    start_logits, end_logits = outputs

    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    if start_index >= end_index:
        return None

    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens).strip()
    return answer if answer else None

def answer_loop(reference: str):
    """Interactive loop: asks user Q:, prints answers A: from reference."""
    while True:
        question = input("Q: ").strip()
        if question.lower() in EXIT_WORDS:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer is None or answer.lower() in ["[cls]", "[sep]"]:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
