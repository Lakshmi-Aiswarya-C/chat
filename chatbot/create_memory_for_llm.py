from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

from pathlib import Path

# Step 1: Load PDFs safely
DATA_PATH = "data/"
def load_all_pdfs_safely(data_path):
    documents = []
    pdf_files = list(Path(data_path).glob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            print(f"Trying to load: {pdf_file}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"✅ Loaded using PyPDFLoader: {pdf_file} ({len(docs)} pages)")
        except Exception as e:
            print(f"⚠️ PyPDFLoader failed for {pdf_file}: {e}")
            try:
                loader = UnstructuredPDFLoader(str(pdf_file))
                docs = loader.load()
                print(f"✅ Loaded using UnstructuredPDFLoader: {pdf_file} ({len(docs)} pages)")
            except Exception as e2:
                print(f"❌ Skipping {pdf_file}: {e2}")
                continue
        documents.extend(docs)
    
    return documents

documents = load_all_pdfs_safely(DATA_PATH)
print(f"\n📚 Total documents loaded: {len(documents)}")

# Step 2: Chunk text
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"✂️ Total text chunks created: {len(text_chunks)}")

# Step 3: Load embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
print("💾 Saving embeddings to FAISS vectorstore...")
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
print("🧠 FAISS index created ✅")

db.save_local(DB_FAISS_PATH)
print("📂 Vectorstore saved locally ✅")


print(f"\n✅ FAISS vector store saved at: {DB_FAISS_PATH}")
