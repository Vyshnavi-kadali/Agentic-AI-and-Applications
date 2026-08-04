# This script performs one-time ingestion of healthcare PDF documents.
# It loads local PDFs, splits them into semantic chunks,
# generates embeddings, and stores them in a FAISS vector database.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Define input data folder and output vector database path
DATA_FOLDER="documents"
VECTOR_DB_PATH="Healthcare_FAISS_Index"

# Validate that the documents folder exists
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(f"Data folder '{DATA_FOLDER}' not found. Please add your PDFs.")

# Step 1: Load PDF Documents

#load PDFs
documents=[]

# Collect all PDF files from the data folder
pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

# Ensure at least one PDF exists
if not pdf_files:
    raise ValueError(f"No PDF files found in '{DATA_FOLDER}'.")

# Load each PDF and extract pages as LangChain documents
for file in pdf_files:
    try:
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
        pdf_docs = loader.load()
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

        # Attach metadata for traceability (source file + page number)
    for i, doc in enumerate(pdf_docs):
            doc.metadata["source"] = file        # PDF filename
            doc.metadata["page"] = i + 1         # Page number starting from 1
            
    documents.extend(pdf_docs)

print(f"loaded {len(documents)} pages from PDFs")


# Step 2: Text Chunking

# Split documents into overlapping chunks to improve retrieval quality
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks=text_splitter.split_documents(documents)
print(f"created {len(chunks)} text chunks")


# Step 3: Generate Embeddings

# Initialize OpenAI embedding model
embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Step 4: Create FAISS Index

# Convert text chunks into vector embeddings and store in FAISS
vectorstore=FAISS.from_documents(chunks, embeddings)

# Save the FAISS index locally for use during chatbot execution
vectorstore.save_local(VECTOR_DB_PATH)

print("Healthcare Knowledge base successfully indexed")