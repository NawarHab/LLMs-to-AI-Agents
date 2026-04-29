from ollama import chat

response = chat(
    model="qwen3.5:0.8b",
    messages=[
        {"role": "user", "content": "Can you explain the quantum physics is 5 lines?"}
    ],
    think=False,  # disables thinking output (for now)
)
# notice your exact answer could be different due to some random processes in the model (learn about that later)
print(response.message.content)
