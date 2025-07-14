# 🔍 Multimodal-RAG-and-Semantic-Search

This repository presents a dual-focused deep learning project covering two advanced areas: a **Multimodal Retrieval-Augmented Generation (RAG) system** and a **Semantic Product Search and Ranking solution**.

The **Multimodal RAG system** is designed to understand and respond to natural language queries based on textual and visual information extracted from PDF documents, leveraging advanced prompting strategies. The **Semantic Product Search and Ranking system** focuses on using deep learning to understand user queries and rank products by semantic relevance, moving beyond traditional keyword-based search.

---

## 🎯 Project Objectives

This project aims to achieve:

* **Multimodal RAG System:** Implementation of a full RAG pipeline capable of processing text and image content from PDF documents, responding to multimodal queries using an integrated LLM, and demonstrating advanced prompting.
* **Semantic Product Search and Ranking (Bonus):** Development of a deep learning-based solution for semantically relevant product retrieval and ranking from a product catalog based on natural language queries.

---
## 🚀 Task 1: Multimodal Retrieval-Augmented Generation (RAG) System

This task focuses on building a robust RAG pipeline capable of understanding and generating responses from both textual and visual information within PDF documents.

### 1.1 Objectives

* Implement a complete RAG pipeline using provided PDF documents containing textual and visual (image-based) information.
* Extract, preprocess, chunk, embed, store, retrieve, and utilize this information to respond to natural language queries through a ChatGPT-like interface.
* Handle both **text-based and image-based queries**, returning contextually relevant responses using an integrated language model.
* Test and demonstrate advanced prompting strategies such as **Chain-of-Thought (CoT)**, **Zero-shot**, and **Few-shot prompting** to enhance reasoning and output quality.

### 1.2 Requirements

* **Data Extraction and Preprocessing:**
    * Parse all three provided PDF files, extracting textual content (financial figures, tabular data, paragraphs) and image content (charts, plots, bar graphs) using OCR and image processing tools.
    * Store each extracted component (text/image with captions or labels) as a separate chunk with metadata.
    * **Recommended Tools:** PyMuPDF, pdfminer.six, Unstructured, pdf2image, Tesseract OCR, EasyOCR, PaddleOCR.

* **Text and Image Embedding:**
    * Convert textual chunks into dense embeddings using encoders like Sentence-BERT or OpenAI Embeddings.
    * Convert visual chunks (images) into embeddings using models like CLIP, BLIP, or other Vision Transformer-based models.
    * Store embeddings in a vector database (FAISS, Chroma, or Weaviate) with proper indexing and metadata tagging.
    * **Recommended Tools:** SentenceTransformers, OpenAI Embeddings, CLIP, BLIP, FAISS, Qdrant, Chroma, Pinecone, Weaviate.

* **Semantic Search and Retrieval:**
    * Implement similarity-based search within the vector database.
    * Enable users to input queries via text or image upload.
    * Retrieve and rank the most relevant chunks (text/image) for a given query, returning results with references to source documents.

* **Language Model Integration:**
    * Integrate an LLM (e.g., LLaMA2, Mistral, GPT-J, GPT-4, Anthropic Claude, Gemini 1.5 Pro via OpenRouter.ai/API).
    * Use retrieved information as context to generate accurate and informative answers.
    * Utilize advanced prompting strategies (CoT, few-shot, zero-shot, LangChain, LlamaIndex, PromptLayer) to improve response quality.

* **User Interface:**
    * Design a simple web-based ChatGPT-like interface using Streamlit or Gradio.
    * Features: text input, image upload, display of search results (ranked chunks, charts, interpreted responses), and option to view source PDF section.
    * The interface must be responsive and intuitive.

* **Evaluation and Visualization:**
    * Visualize embedding space (TSNE/PCA plots) and search results.
    * Display retrieval hit rates and measure semantic similarity scores.
    * Evaluate final responses using metrics: Relevance score (manual/cosine similarity), BLEU/ROUGE for generative quality, and query response time.

* **Documentation:**
    * Provide a detailed technical report (LaTeX, Springer LNCS format) covering system architecture, component explanation, evaluation, prompt engineering examples, and challenges.
    * Submit a prompt log (`.txt`) with all prompts used.
    * Ensure reproducibility of the entire pipeline.

### 1.3 Expected Outcomes

* A fully functional RAG system capable of handling multimodal queries.
* Demonstration of reasoning and coherence in LLM responses via advanced prompt strategies.
* Visualization of embeddings and retrieval operations.
* A deployable web interface with working semantic search.

---

## ✨ Bonus Task 2: Semantic Product Search and Ranking

This bonus task explores the application of deep learning for understanding user intent in product search and ranking products by semantic relevance.

### 2.1 Objectives

* Develop a deep learning-based solution for **semantic product search and ranking**.
* Accept a natural language query, retrieve candidate products from a catalog, and rank them according to semantic relevance.
* Train the model on query-product pairs and deploy it as a web application that serves search results in real-time.
* Incorporate a modern text representation method (e.g., word embeddings, BERT) to encode queries and product information, learning relevance scores.

### 2.2 Requirements

* **Data Preparation:**
    * Load the Amazon Shopping Queries Dataset ([Link](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset)).
    * Combine product title and product description columns for rich product representation.
    * Apply text preprocessing: lowercase conversion, stop word removal, lemmatization/stemming, special character removal.
    * Split dataset into training, validation, and test sets (e.g., 70%-15%-15%).

* **Text Representation:**
    * Transform text data into numerical representations using: TF-IDF, Word embeddings (Word2Vec, GloVe, FastText), or Pretrained models (BERT, GPT).
    * **Recommendation:** Consider transformer-based models like BERT for enhanced semantic understanding.

* **Model Training and Evaluation:**
    * Train a suitable deep learning model to learn relevance between queries and products.
    * Fine-tune model hyperparameters to optimize performance.
    * Visualize training and validation loss curves.
    * Evaluate the model using appropriate ranking metrics: **NDCG, MAP, Precision@K, Recall@K, and F1@K**.

* **Deployment:**
    * Create a web interface with a simple text box for user queries.
    * The query should be passed to the trained model, which returns and displays a ranked list of relevant products.
    * Utilize both product title and product description fields during training and inference.

---
