from vidavox.core import RAG_Engine
from vidavox.generation.llm import Client
from vidavox.utils.word_extractor import extract_keywords

file_paths = ["./unrelated/Draft_POD.md"]
file_dir = "./unrelated"
query = "Di buku POD isinya apa saja?"

keywords = extract_keywords(query)

result = RAG_Engine().from_directory(file_dir).query(query_text=query, keywords=keywords)

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
# openai_client = Client(model="openai:gpt-3.5-turbo")
lamma_client = Client(model="ollama:llama3.2:1b")

# response_from_open_ai = openai_client.chat.completions.create(messages=messages, temperature=0.75)
response_from_llama = lamma_client.chat.completions.create(messages=messages, temperature=0.75)

print('========== OPENAI ==========')
# print(response_from_open_ai.choices[0].message.content)
print('\n')
print('========== OLLAMA ==========')
print(response_from_llama.choices[0].message.content)





