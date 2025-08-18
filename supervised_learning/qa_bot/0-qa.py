#!/usr/bin/env python3
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer

# Load once (so it doesn’t reload for each call)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

def question_answer(question: str, reference: str) -> str:
    """
    Finds a snippet of text within reference that answers the question.
    Uses bert-uncased-tf2-qa model and bert-large-uncased-whole-word-masking-finetuned-squad tokenizer.
    """

    # Encode inputs
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

    # Get most likely start and end token positions
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    if start_index >= end_index:
        return None

    # Convert tokens back to string
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)

    # Clean up formatting (e.g., ## in wordpieces)
    return answer.strip() if answer else None
