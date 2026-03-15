from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "What was the original name of Microsoft before it became Microsoft?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.2  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query : {query}")

print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")