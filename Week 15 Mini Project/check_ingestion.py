# check_ingestion.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FOLDER = "documents"

# Configure your text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # same as your ingestion
    chunk_overlap=300
)

pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

total_pages = 0
total_chunks = 0

for pdf_file in pdf_files:
    path = os.path.join(DATA_FOLDER, pdf_file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    total_pages += len(pages)
    
    chunks = text_splitter.split_documents(pages)
    total_chunks += len(chunks)
    
    print(f"PDF: {pdf_file} | Pages: {len(pages)} | Chunks: {len(chunks)}")

print("-"*60)
print(f"Total PDFs: {len(pdf_files)}")
print(f"Total Pages: {total_pages}")
print(f"Total Chunks: {total_chunks}")