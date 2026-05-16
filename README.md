# LegaLease - Legal AI Assistant

LegaLease is an intelligent, RAG-based (Retrieval-Augmented Generation) legal AI assistant tailored for rental-law guidance, lease summarization, and clause risk checks in one clean workspace. The system leverages locally hosted Large Language Models (LLMs) via Ollama, ensuring data privacy and fast inference.

## Features

The application provides a Streamlit-based user interface with three core functionalities:

1. **Legal Chat (Q&A System)**
   - Ask complex legal queries regarding rental laws and agreements.
   - Utilizes FAISS for efficient retrieval of relevant legal chunks from your custom knowledge base.
   - Generates grounded, accurate answers using local LLMs with a fallback mechanism if generation fails.
   - Provides full transparency by displaying the source document chunks used to generate the answer.

2. **Lease Summarizer**
   - Instantly extracts key lease fields including **Rent, Notice Period, Deposit, and Maintenance**.
   - Powered by LLM extraction with a robust regex-based fallback to ensure high reliability even if the model struggles with formatting.

3. **Clause Checker (Risk Analysis)**
   - Paste lease agreements or individual clauses to identify potential risks.
   - Flags clauses as **High, Medium, or Low** risk based on rental law implications (e.g., immediate eviction, no refund, excessively high deposits).
   - Provides clear reasoning for why a clause was flagged, acting as a first line of defense before signing agreements.

## Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Local LLM Backend:** [Ollama](https://ollama.com/) (using `llama3` by default)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (CPU-optimized)
- **Data Processing:** NLTK for sentence tokenization, NumPy
- **Testing:** `pytest`

## Project Structure

```
LegalEaseRAG/
├── app/
│   ├── main.py              # Streamlit frontend UI
│   ├── rag_engine.py        # Core RAG logic, summarization, and risk checking
│   ├── retriever.py         # FAISS retrieval and context ranking
│   ├── embeddings.py        # Document embedding generation
│   └── llm.py               # REST API wrapper for local Ollama instance
├── config/
│   └── settings.py          # Centralized configuration (loads from .env)
├── data/                    # Directory for raw .txt files and FAISS indices
├── scripts/
│   └── ingest.py            # Script to chunk, embed, and index legal documents
├── tests/                   # Pytest suite for core components
├── requirements.txt         # Python dependencies
└── .env.example             # Example environment variables
```

## Prerequisites

1. **Python 3.9+** installed on your system.
2. **Ollama** installed and running locally.
   - Download Ollama from [ollama.com](https://ollama.com/).
   - Pull the Llama 3 model (or your preferred model):
     ```bash
     ollama pull llama3
     ```
   - Ensure the Ollama server is running:
     ```bash
     ollama serve
     ```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd LegalEaseRAG
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` if you wish to change the default models (`OLLAMA_MODEL=llama3`, `EMBEDDING_MODEL=all-MiniLM-L6-v2`).

## Data Ingestion

Before using the Legal Chat functionality, you need to build the knowledge base.

1. Place your plain text legal documents (`.txt` format) inside the `data/` directory.
2. Run the ingestion script to chunk the text, generate embeddings, and build the FAISS index:
   ```bash
   python scripts/ingest.py
   ```
   *Note: This will generate `faiss_index.bin` and `chunks.json` in the `data/` directory.*

## Usage

Start the Streamlit application:

```bash
streamlit run app/main.py
```

The application will launch in your default web browser at `http://localhost:8501`. From there, you can navigate between the Legal Chat, Lease Summarizer, and Clause Checker tabs.

## Testing

The project includes a comprehensive test suite covering embeddings, prompt building, retrieval, and the RAG engine itself. 

To run the tests, ensure your virtual environment is activated and run:
```bash
pytest
```
