from vidavox.core import Retrieval_Engine
from vidavox.utils.word_extractor import extract_keywords
from vidavox.settings import DatabaseSettings
from vidavox.document_store.store import VectorStorePsql

import json

# === 1. Define your file paths / directory ===
file_paths = ["./unrelated/Journal.pdf"]
file_dir = "./unrelated"

# === 2. Your query / keywords setup ===
query_single = "What is the introduction of this research paper?"
keywords_single = []  # no keywords for the single‐query example

# Later, for batch:
query_batch = [
    "What is the introduction of this research paper?",
    "introduction of the research paper"
]
keywords_batch = []  # will fill this in below by calling extract_keywords()

# === 3. Set up your database / vector store ===
db_settings = DatabaseSettings(
    url="postgresql+asyncpg://postgres:admin@localhost:5432/test_db",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)
vector_store = VectorStorePsql(db_settings=db_settings)

# === 4. Instantiate the RetrievalEngine (with vector_store) & index your files ===
engine = Retrieval_Engine(
    embedding_model="Snowflake/snowflake-arctic-embed-l-v2.0",
)
# This will load/index all PDFs in file_paths into your vector store
engine.from_paths(file_paths)

# === 5. Run a single‐query retrieval (no keywords) ===
print("--- Retrieving (single) without keywords ---")
result_single_no_kw = engine.retrieve(
    query_text=query_single,
    keywords=keywords_single,
    top_k=9,
    threshold=0.3
)
print(json.dumps(result_single_no_kw, indent=2))

# === 6. (Optionally) run a single‐query retrieval WITH keywords ===
# If you wanted to supply your own keywords for that one query:
keywords_single = extract_keywords(query_single, top_n=3)
result_single_with_kw = engine.retrieve(query_text=query_single,
                                        keywords=keywords_single,
                                        top_k=9, threshold=0.3)
print(json.dumps(result_single_with_kw, indent=2))

# === 7. Prepare keywords for the batch queries ===
for q in query_batch:
    # each element in keywords_batch is a small list of top‐3 keywords for that query
    keywords_batch.append(extract_keywords(q, top_n=3))
print(f"Extracted Keywords for batch: {keywords_batch}")

# === 8. Run the batch retrieval (without and then with keywords) ===
print("\n--- Retrieving (batch) without keywords ---")
result_batch_no_kw = engine.retrieve_batch(
    queries=query_batch,
    keywords=[[] for _ in query_batch],  # empty keywords for each query
    top_k=9,
    threshold=0.3
)
print(json.dumps(result_batch_no_kw, indent=2))

print("\n--- Retrieving (batch) with keywords ---")
result_batch_with_kw = engine.retrieve_batch(
    queries=query_batch,
    keywords=keywords_batch,
    top_k=9,
    threshold=0.3
)
print(json.dumps(result_batch_with_kw, indent=2))

# === 9. Save all four outputs to distinct JSON files ===
print("\n--- Saving results to files ---")

# Single (no keywords)
with open("result_single_without_keywords.json", "w", encoding="utf-8") as f:
    json.dump(result_single_no_kw, f, ensure_ascii=False, indent=2)
print("Saved single‐query result (no keywords) to 'result_single_without_keywords.json'")

# (Uncomment if you run the single‐with‐keywords)
with open("result_single_with_keywords.json", "w", encoding="utf-8") as f:
    json.dump(result_single_with_kw, f, ensure_ascii=False, indent=2)
print("Saved single‐query result (with keywords) to 'result_single_with_keywords.json'")

# Batch (no keywords)
with open("result_batch_without_keywords.json", "w", encoding="utf-8") as f:
    json.dump(result_batch_no_kw, f, ensure_ascii=False, indent=2)
print("Saved batch‐query result (no keywords) to 'result_batch_without_keywords.json'")

# Batch (with keywords)
with open("result_batch_with_keywords.json", "w", encoding="utf-8") as f:
    json.dump(result_batch_with_kw, f, ensure_ascii=False, indent=2)
print("Saved batch‐query result (with keywords) to 'result_batch_with_keywords.json'")
