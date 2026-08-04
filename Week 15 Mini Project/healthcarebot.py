# This script runs the Healthcare RAG Chatbot.
# It loads the FAISS vector database, retrieves relevant document chunks,
# and generates grounded responses using an LLM with conversation memory.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# ENVIRONMENT SETUP

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

VECTOR_DB_PATH = "Healthcare_FAISS_Index"

# Initialize conversation memory (session-based)
# Stores chat history for context-aware follow-up questions
memory = ConversationBufferMemory(
    return_messages=True
)


# LOAD VECTOR DATABASE

# Initialize embedding model (must match ingestion model)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load precomputed FAISS index from local storage
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True  # Required for local FAISS loading
)

# Configure retriever to fetch top-4 most relevant document chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# LLM CONFIGURATION

# Initialize chat model with deterministic output (temperature=0)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# PROMPT TEMPLATE (HALLUCINATION CONTROL)

# Custom prompt ensures:
# - Document-grounded answers
# - Proper citation format
# - No external knowledge usage
# - Safe fallback response if information is missing

prompt = ChatPromptTemplate.from_template("""
You are a Healthcare Information Assistant.

Previous Conversation:
{history}

Your role:
- Provide accurate health-related information using ONLY the provided healthcare documents.
- Include a citation for each piece of information in this format: (Document Name, Page X).
- Do NOT generate medical advice beyond the provided documents.
- Do NOT add information from outside knowledge.

If the answer is not found in the provided documents, respond with:
"I’m not sure based on the available healthcare documents."

Guidelines:
- Be clear and easy to understand.
- Use simple, patient-friendly language.
- Keep responses concise unless detailed explanation is requested.
- Do not mention the word "context" in your response.
- Do not guess or assume information.

Healthcare Documents:
{context}

User Question:
{question}

Answer:
""")


# CHAT LOOP WITH ERROR HANDLING
def chat():
    print("\nHealthcare Assistant (type 'exit' to quit)\n")

    while True:
        question = input("User: ")
        # Exit condition
        if question.lower() == "exit":
            break
        # Step 1: Retrieve Documents
        # --- Document retrieval with error handling ---
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"\nHealthcare Assistant: Error retrieving documents. ({e})")
            print("-" * 60)
            continue
        # Handle case where no relevant documents are found
        if not docs:
            print("\nHealthcare Assistant: I don’t have enough information in the provided documents.")
            print("-" * 60)
            continue

        # Step 2: Generate Response
        
        # --- Response generation with error handling ---
        try:
            # Embed citations from metadata into context
            context = "\n\n".join(
                f"{doc.page_content}\n(Source: {doc.metadata.get('source','Unknown')}, Page {doc.metadata.get('page','N/A')})"
                for doc in docs
            )
            
            # Retrieve chat history from memory
            chat_history = memory.load_memory_variables({})["history"]
            # Convert structured messages into readable string format
            chat_history_str = "\n".join(
                f"User: {m.content}" if m.type == "human" else f"Assistant: {m.content}"
                for m in chat_history
            )

            # Invoke LLM with augmented prompt (history + retrieved documents)
            response = llm.invoke(
                prompt.format_messages(
                    history=chat_history_str,
                    context=context,
                    question=question
                )
            )

            print("\nHealthcare Assistant:", response.content)
            print("-" * 60)

            # Save conversation to memory
            memory.save_context(
                {"input": question},
                {"output": response.content}
            )

        except Exception as e:
            print(f"\nHealthcare Assistant: An error occurred while generating the response. ({e})")
            print("-" * 60)
        

# Entry point of the script
if __name__ == "__main__":
    chat()
