# 📚 Vidavox RAG Pipeline

![Vidavox Banner](https://user-images.githubusercontent.com/your-image-placeholder/banner.png)

**Vidavox RAG Pipeline** is a framework that helps developers build Retrieval-Augmented Generation (RAG) applications with ease. Forget the painful trial-and-error process—this pipeline simplifies RAG development and offers a powerful foundation for various use cases.

## 🚀 Why Vidavox RAG?

Many businesses and developers fail when attempting to build RAG systems due to the complexity and hidden factors that affect success. **Vidavox RAG Pipeline** is designed to reduce the friction, allowing you to:

- Focus on your use case without reinventing the wheel
- Collaborate and learn from shared experiences
- Quickly iterate and optimize your RAG solution

### 🌍 Target Audience

**AI Developers** interested in building or integrating RAG-based solutions.

## ✨ Key Features

- **Document Parsing:** Supports multiple file formats (PDF, DOCX, TXT, CSV, XLS, MD)
- **Retrieval:** Recursive chunking, support for various embedding models (Sentence Transformers), and dense/sparse vector search
- **Generation:**
  - Open-source LLMs through **Ollama**
  - Proprietary models like **GPT-4**

## 📥 Setup and Installation

1. Install Vidavox RAG using pip:
   ```bash
   pip install vidavox
   ```

2. To work with open-source models, install Ollama and download the desired model.
   Refer to [Ollama Installation Guide](link-to-ollama-guide) for details.

## 📖 Usage Example

Here's how to get started with Vidavox RAG:

```python
from vidavox.core import RAG_Engine
from vidavox.generation.llm import Client
import os

file_paths = ["./unrelated/Draft_POD.md"]  # Ensure the file path is correct
for file_path in file_paths:
    print(f"Checking {file_path}: {os.path.isfile(file_path)}")

engine = RAG_Engine(use_async=True).from_documents(file_paths)
query = "Di buku POD isinya apa saja?"
result = engine.query(query_text=query)

rag_prompt = """
### Context (Retrieved Documents):
{context}

### User Question:
{question}

### Instructions:
1. **Identify Relevant Information:** Extract key facts and details from the context.
2. **Reasoning & Inference:** If explicit, provide a concise answer; otherwise, infer the response.
3. **Uncertainty Handling:** State "I cannot provide a definitive answer" if the data is insufficient.
4. **Structured Output:** Provide a clear, well-structured response.
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": rag_prompt.format(context=result, question=query)},
]

openai_client = Client(model="openai:gpt-3.5-turbo")
ollama_client = Client(model="ollama:llama3.2:1b")

response_openai = openai_client.chat.completions.create(messages=messages, temperature=0.75)
response_ollama = ollama_client.chat.completions.create(messages=messages, temperature=0.75)

print('========== OPENAI ==========')
print(response_openai.choices[0].message.content)
print('\n========== OLLAMA ==========')
print(response_ollama.choices[0].message.content)
```

## ⚙️ Configuration

### Embedding Model Configuration

By default, `RAG_Engine().from_documents(file_paths)` uses `all-MiniLM-L6-v2` as the embedding model.
You can specify a different embedding model like this:

```python
engine = RAG_Engine(embedding_model='Snowflake/snowflake-arctic-embed-l-v2.0').from_documents(file_paths)
```

### Querying with Keywords

Enhance your query by extracting and passing keywords:

```python
keywords = extract_keywords(query)
result = engine.query(query_text=query, keywords=keywords)
```

### LLM Client Configuration

1. Ensure you have an `OPENAI_API_KEY` in your `.env` file or system environment.
2. To use Ollama, install and download the model, then:
   ```python
   from vidavox.generation.llm import Client
   client = Client(model="ollama:llama3.2:1b")
   ```

## 📊 Evaluation Results

### How Does Model Size Affect Dependency on Embedding Quality for RAG Accuracy?

We conducted a pre-test to compare different LLM sizes and embedding models on RAG performance.

#### Methods

- **Dataset:** Real-world data from a specific domain in Bahasa Indonesia
- **Query-Context Pairs:** Generated using Llama 3.3 70B 4Bit and filtered based on groundedness, relevance, and non-explicitness
- **Embedding Models Compared:**
  - all-MiniLM-L6-v2: Small, efficient, low computational cost
  - Snowflake/snowflake-arctic-embed-l-v2.0: Larger size, strong multilingual performance
- **LLM Comparison:**
  - Llama 3.1 8B Instruct vs. Llama 3.2 1B Instruct
- **Evaluation:** Llama 3.3 70B served as the LLM-judge to assess accuracy

![Alt text](/images/rag_eval.png)

#### Results

- Llama 3.1 8B consistently outperformed Llama 3.2 1B
- Higher quality embeddings improved accuracy across the board
- Smaller models are more dependent on embedding quality for good accuracy

#### Takeaway

For the best accuracy:
- Larger LLMs + High-Quality Embeddings are recommended
- For resource-constrained applications:
  - Embedding quality becomes the key factor for smaller models

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork this repository
2. Create a new branch (feature/your-feature-name)
3. Submit a pull request with a clear description of your changes

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

For support or collaboration, reach out to:
📧 ariansyah@vidavox.ai

## 🌟 Acknowledgments

Special thanks to the open-source community and contributors who made this project possible.