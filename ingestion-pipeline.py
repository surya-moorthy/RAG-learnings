import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore

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

def main():
    print("Main functions")

    documents = load_documents(docs_path="docs")


if __name__ == "__main__":
    main()