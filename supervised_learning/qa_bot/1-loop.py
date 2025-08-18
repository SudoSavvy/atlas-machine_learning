#!/usr/bin/env python3

def chat():
    while True:
        question = input("Q: ").strip()
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            # You can replace this with any logic for answering
            print("A:", "I heard you say:", question)

if __name__ == "__main__":
    chat()
