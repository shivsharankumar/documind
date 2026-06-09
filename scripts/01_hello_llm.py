# scripts/01_hello_llm.py
from documind.llm import LLM, Message

llm = LLM()
resp = llm.complete(
    [
        Message(
            "system",
            "You are DocuMind, a precise technical assistant. "
            "If unsure, say so. Never fabricate code or APIs.",
        ),
        Message("user", "In one sentence, what is a vector database?"),
    ]
)
print(resp.text)
print(f"tokens: in={resp.input_tokens} out={resp.output_tokens}")
