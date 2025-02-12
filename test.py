from vidavox.core import RAG_Engine, extract_keywords
from vidavox.generation.llm import Client

file_paths = ["./Draft_POD.md"]
import os
for file_path in file_paths:
    print(f"Checking {file_path}: {os.path.isfile(file_path)}")

engine = RAG_Engine(use_async=True).from_documents(file_paths)

query = "Buku Usulan POD terdiri dari?"

result = engine.query(query_text=query, keywords=extract_keywords(query))

rag_prompt = """
    ### Context (Retrieved Documents):
    {context}

    ### User Question:
    {question}

    ### Instructions:
    1. **Identify Relevant Information:** Extract key facts and details from the retrieved context that are directly related to the user's question.
    2. **Reasoning & Inference:** If the answer is explicitly stated, provide it concisely. If implicit, use logical reasoning to infer a response.
    3. **Uncertainty Handling:** If the information is insufficient or ambiguous, state: "Based on the available data, I cannot provide a definitive answer."
    4. **Structured Output:** Provide the answer in a clear, well-structured format.
    """
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": rag_prompt.format(context=result, question=query)},
]
client = Client(model="openai:gpt-3.5-turbo")

response = client.chat.completions.create(messages=messages, temperature=0.75)
print(response.choices[0].message.content)


