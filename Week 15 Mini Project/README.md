## Project Overview

This project focuses entirely on the healthcare domain and demonstrates a Healthcare RAG-based Chatbot named "HEALTHCARE_CHATBOT". The system is designed to answer user queries related to various healthcare topics using document-grounded knowledge.

The chatbot can:

Answer disease-related questions such as HIV, Diabetes, and Cardiovascular conditions

Provide information about nutritional guidance and recommended foods

Respond to queries about hospital facilities available at TXH hospitals

Handle FAQs sourced from the official TXH hospital website, enabling the bot to behave like an integrated TXH healthcare assistant

Provide information about applicable government healthcare schemes (e.g., Andhra Pradesh Arogyasri) that may help users financially based on their scenario

The chatbot strictly answers questions only based on the available documents used during ingestion.
If the requested information is not found in the documents, the bot responds with:

“I'm not sure based on the available documents.”

This ensures reliability, prevents hallucination, and maintains document-grounded responses.

## Key Features
### PDF-Based Knowledge Base
Uses curated healthcare documents (diseases, nutrition, TXH FAQs, vaccines, and government schemes) as the authoritative source of information.

### Retrieval-Augmented Generation (RAG)
Converts user queries into embeddings, retrieves relevant document chunks using FAISS, and generates grounded responses with gpt-4o-mini.

### Context-Aware Conversations
Supports follow-up questions using ConversationBufferMemory, maintaining session-level chat history.

### Source Document Citation
Each response includes the originating document name for transparency and traceability.

### Hallucination Control
If information is not found in the documents, the chatbot responds:

“I'm not sure based on the available documents."

# Architecture
User Question
      ↓
Embedding (text-embedding-3-small)
      ↓
Vector Similarity Search (FAISS)
      ↓
Relevant Healthcare Document Chunks
      ↓
Prompt Augmentation
      ↓
LLM Answer (gpt-4o-mini)

## Flow Explanation:

User Question – The user asks a healthcare-related question.

Embedding – Converts the question into a vector representation.

Vector Similarity Search (FAISS) – Retrieves the most relevant chunks from precomputed document embeddings.

Prompt Augmentation – Retrieved chunks are appended to the prompt for better contextual answers.

LLM Answer – Generates a precise answer using the augmented prompt.

## Tech Stack

* **Python**
* **OpenAI API**

  * `gpt-4o-mini` (chat model)
  * `text-embedding-3-small` (embedding model)
* **LangChain**
* **FAISS** (vector database)
* **PyPDF** (PDF ingestion)

# Project Structure

```text
RAG_Project/
│
├── documents/
│   ├── Diabetes.pdf
│   ├── HIV and AIDS.pdf
│   ├── Animal bites.pdf
│   ├── TXH Healthcare Queries.pdf
│   ├── Andhra Pradesh Arogyasri.pdf
│   ├── Nutrition Guide.pdf
│   ├── Vaccines and Immunization.pdf
│   └── Cardiovascular diseases.pdf
│
├── ingest.py
├── healthcarebot.py
├── Healthcare_FAISS_Index/
├── check_ingestion.py
├── requirements.txt
└── README.md
```

---

# Public data source links
TXH Hospitals: https://txhospitals.in/faqs/

Arogyasri Scheme: https://drntrvaidyaseva.ap.gov.in/documents/d/guest/138

HIV & AIDS: https://www.who.int/news-room/fact-sheets/detail/hiv-aids

Animal Bites: https://www.who.int/news-room/fact-sheets/detail/animal-bites

Cardiovascular Diseases: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

Diabetes: https://www.who.int/news-room/fact-sheets/detail/diabetes

Vaccines & Immunization: https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
```

## PDF Ingestion (One-Time Step)

This step converts Healthcare PDFs into vector embeddings.

```bash
python ingest.py
```

What this does:

* Loads all PDFs from `documents/`
* Splits them into semantic chunks
* Generates embeddings using OpenAI
* Stores vectors locally using FAISS

---

## Running the Chatbot
```bash
python healthcarebot.py
```

## Example questions
1. What are the symptoms of HIV?

2. What are the types of diabetes?

3. Does TXH hospital provide ambulance facility?

4. Is Arogyasri available for cardiac treatment?

5. What food should a diabetes patient take?

## Memory Handling

The chatbot uses ConversationBufferMemory from LangChain to store chat history within the current session.

The conversation history is maintained only until the session expires.

The bot supports context-aware follow-up questions.

Retrieval is performed using relevant document chunks while incorporating previous conversational context.

The model is explicitly prompted to avoid hallucination and respond with “I'm not sure based on the provided documents” when information is unavailable.

## Design Principles
1. Document as Source of Truth

Healthcare PDFs are the authoritative knowledge base.

2. Retrieval Before Generation

The system retrieves relevant document chunks before generating responses.

3. Transparency Through Citations

Each answer includes source document references.

4. Safety-First Healthcare Design

The chatbot refuses to guess when information is missing.

5. Modular & Extensible Architecture

Clear separation between ingestion and chatbot logic.

## Limitations

No role-based access control

Local FAISS storage (not distributed)

Session-based memory only

Not a replacement for professional medical consultation
