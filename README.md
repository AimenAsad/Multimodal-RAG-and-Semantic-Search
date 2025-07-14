# RAG: Multimodal Retrieval-Augmented Generation System

A comprehensive **Retrieval-Augmented Generation (RAG) system** that supports both **text and image inputs**, built with **Streamlit**, the free-preview **Gemini 1.5 Pro API**, and **LangChain**.

---

## ✨ Features

* **Multimodal Support**: Seamlessly processes both text and image queries, enabling a richer interaction experience.
* **PDF Processing**: Efficiently extracts and chunks text from PDF documents for indexing and retrieval.
* **Vector Storage**: Utilizes **ChromaDB** to store and retrieve embeddings, facilitating fast and relevant context lookup.
* **LLM Integration**: Leverages the powerful **Gemini 1.5 Pro API** (via its OpenAI-compatible endpoint) for generating coherent and contextually relevant responses.
* **Evaluation Module**:
    * **BLEU and ROUGE metrics** for quantitative assessment of response quality.
    * **Visualization of embedding spaces** to understand the semantic relationships within the data.
    * **Predefined test questions** for automated and consistent evaluation of the system's performance.
* **Streamlit Interface**: Provides an intuitive and user-friendly web application:
    * Features a wide, two-pane layout with custom theming for optimal usability.
    * **Document management** and **query controls** are conveniently located on the left pane.
    * **Answer display** and **source references** are presented clearly on the right pane.

---

## 🏗️ Project Structure

The project is organized into logical directories to ensure maintainability and clarity:

```text
multimodal-rag-system/
├── src/
│   ├── app.py                  # Main Streamlit application entry point
│   ├── rag/                    # Core RAG components for embeddings, vector DB, and retrieval
│   │   ├── embeddings.py       # Handles multimodal embedding generation
│   │   ├── vector_db.py        # Interface for ChromaDB operations
│   │   └── retrieval.py        # Manages the retrieval pipeline for context
│   ├── llm/
│   │   └── generation.py       # Wrapper for interacting with the Gemini 1.5 Pro API
│   ├── processing/
│   │   └── pdf_processor.py    # Logic for PDF text extraction and intelligent chunking
│   └── evaluation/
│       ├── metrics.py          # Implements BLEU, ROUGE, and latency calculation
│       └── visualisation.py    # Scripts for plotting embedding spaces (e.g., t-SNE, PCA)
├── data/                       # Directory to store PDF files for ingestion into the RAG system
├── model_cache/                # Local cache for downloaded sentence-transformers models
├── .streamlit/                 # Streamlit specific configuration files
│   ├── config.toml             # Defines Streamlit app layout and custom theme settings
│   └── secrets.toml            # Stores sensitive information like GEMINI_API_KEY securely
├── tests/                      # (Optional) Contains unit and integration tests for various modules
└── requirements.txt            # Lists all Python dependencies required to run the project
