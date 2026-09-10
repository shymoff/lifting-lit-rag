"""Sanity check: confirm Ollama is reachable and the generator model responds."""

import time

import ollama

GENERATOR_MODEL = "qwen2.5:3b-instruct-q4_K_M"
PROMPT = "In one sentence, what is progressive overload?"


def main() -> None:
    start = time.perf_counter()
    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.perf_counter() - start

    print(f"Model: {GENERATOR_MODEL}")
    print(f"Prompt: {PROMPT}")
    print(f"Response: {response['message']['content']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
