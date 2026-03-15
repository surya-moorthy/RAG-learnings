import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all the files from the docs directory"""
    print(f"Loading documents from {docs_path}....")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist, Please create it and add your company files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"can't able to find .txt in {docs_path}...")
    
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        print(f"Sources : {doc.metadata['source']}")
        print(f"Content length : {len(doc.page_content)} characters")
        print(f"Content preview : {doc.page_content[:100]}")
        print(f"Metadata : {doc.metadata}\n\n")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"------Chunk {i + 1}--------")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)
        if len(chunks) > 5:
            print(f"\n------- and {len(chunks) - 5} more chunks.....")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings  and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

    print("---Creating vector store---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vector_store

def main():
    print("Main functions")

    # loading documents
    documents = load_documents(docs_path="docs")

    # chunking documents , which splits the docs into chunks
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)

    # creating vector embeddings and stored it in a folder 
    vector_store = create_vector_store(chunks=chunks, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()