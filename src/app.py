import torch
torch.classes.__path__ = []
import streamlit as st
from PIL import Image
import os
import time
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(
    page_title="RAG",
    page_icon="📚",
    layout="wide"
)
#from src.config.settings import settings
from src.rag.embeddings import MultimodalEmbedder
from src.rag.vector_db import VectorStore
from src.rag.retrieval import Retriever
from src.llm.generation import ResponseGenerator
from src.processing.pdf_processor import PDFProcessor
from src.evaluation.visualisation import plot_embeddings, plot_search_results
from src.evaluation.metrics import Evaluator

st.markdown(
    """
    <style>
      /* hide the default Streamlit header/footer */
      header, footer { visibility: hidden; }

      /* App header bar */
      .app-header {
        background-color: #1F4E79;
        padding: 1rem;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
      }

      /* Left pane card */
      .left-pane {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }

      /* Right pane card */
      .right-pane {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }

      /* Response box */
      .response-box {
        background: #EAF4FC;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
      }

      /* Source content */
      .source-content {
        font-size: 0.9rem;
        line-height: 1.4;
      }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="app-header">📚 RAG</div>', unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """Initialize components without UI elements"""
    embedder = MultimodalEmbedder()
    vector_db = VectorStore(embedder)
    retriever = Retriever(vector_db, embedder)
    llm = ResponseGenerator()
    return embedder, vector_db, retriever, llm

def handle_pdf_processing(vector_db):
    """Separate function for PDF processing with UI"""
    if vector_db.text_collection.count() == 0 and vector_db.image_collection.count() == 0:
        with st.spinner("Processing initial PDFs..."):
            pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
            progress_bar = st.progress(0)
            
            for i, pdf_file in enumerate(pdf_files):
                try:
                    processor = PDFProcessor(os.path.join("data", pdf_file))
                    chunks = processor.process()
                    vector_db.add_documents(chunks)
                    progress_bar.progress((i+1)/len(pdf_files))
                except Exception as e:
                    st.error(f"Failed to process {pdf_file}: {str(e)}")
            progress_bar.empty()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data/ directory"""
    file_path = os.path.join("data", uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_path

def main():
    # Initialize components & process any existing PDFs
    embedder, vector_db, retriever, llm = init_system()
    handle_pdf_processing(vector_db)

    # Two-column split: left for docs + query, right for results
    left_col, right_col = st.columns([1, 2], gap="medium")

    with left_col:
        st.markdown('<div class="left-pane">', unsafe_allow_html=True)
        st.header("Docs")
        st.metric("Text chunks", vector_db.text_collection.count())
        st.metric("Image chunks", vector_db.image_collection.count())

        uploaded = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True
        )
        if uploaded:
            for f in uploaded:
                path = save_uploaded_file(f)
                chunks = PDFProcessor(path).process()
                vector_db.add_documents(chunks)
                st.success(f"Added {f.name}")

        st.markdown("---")
        st.header("Query")
        query_tab, image_tab = st.tabs(["Text", "Image"])
        with query_tab:
            query_text = st.text_input("Enter your question")
        with image_tab:
            uploaded_image = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        if st.button("🔎 Search"):
            if not (query_text or uploaded_image):
                st.warning("Please enter text or upload an image")
            else:
                with st.spinner("Analyzing…"):
                    if query_text:
                        context = retriever.retrieve(query_text)
                        prompt = query_text
                    else:
                        img = Image.open(uploaded_image)
                        context = retriever.retrieve(img)
                        prompt = "Explain this image"

                    response = llm.generate_response(prompt, context)

                st.subheader("Answer")
                st.markdown(
                    f'<div class="response-box">{response}</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Sources")
                for i, chunk in enumerate(context[:3], start=1):
                    with st.expander(
                        f"Source {i}: {chunk['metadata']['source']} (p.{chunk['metadata']['page']})"
                    ):
                        if chunk["metadata"]["type"] == "image":
                            st.image(chunk["metadata"]["image_path"])
                        st.markdown(
                            f'<div class="source-content">{chunk["content"]}</div>',
                            unsafe_allow_html=True
                        )
        st.markdown('</div>', unsafe_allow_html=True)
            
eval_tab, vis_tab = st.tabs(["Evaluation Metrics", "Visualization"])

with vis_tab:
    st.header("Visualization")
    
    # Initialize system components
    _, vector_db, _, _ = init_system()
    
    try:
        # Initialize empty arrays with proper dimensions
        text_embeddings = np.empty((0, 384))  # Sentence-BERT dimension
        image_embeddings = np.empty((0, 512))  # CLIP dimension
        text_labels = []
        image_labels = []

        # Load text embeddings
        if vector_db.text_collection.count() > 0:
            text_data = vector_db.text_collection.get(include=["embeddings"])
            if text_data["embeddings"]:
                text_embeddings = np.array(text_data["embeddings"])
                text_labels = [f"Text: {m['source']} (p.{m['page']})" 
                             for m in vector_db.text_collection.get()["metadatas"]]

        # Load image embeddings
        if vector_db.image_collection.count() > 0:
            image_data = vector_db.image_collection.get(include=["embeddings"])
            if image_data["embeddings"]:
                image_embeddings = np.array(image_data["embeddings"])
                image_labels = [f"Image: {m['source']} (p.{m['page']})" 
                              for m in vector_db.image_collection.get()["metadatas"]]

        # Create visualization
        if text_embeddings.size > 0 or image_embeddings.size > 0:
            fig = plot_embeddings(text_embeddings, image_embeddings, 
                                 text_labels, image_labels)
            st.plotly_chart(fig)
        else:
            st.warning("Process documents to enable visualization")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

# Modified Evaluation Tab
with eval_tab:
    st.header("System Evaluation Metrics")
    
    if st.button("Run Comprehensive Evaluation"):
        # Reinitialize system components
        _, _, retriever, llm = init_system()
        evaluator = Evaluator(retriever, llm)
        
        try:
            with st.expander("Performance", expanded=True):
                test_queries = [
                    ("What was 2023 revenue?", "financials.pdf"),
                    ("Explain FYP requirements", "FYPHandbook2023.pdf")
                ]
                hit_rate = evaluator.calculate_hit_rate(
                    [q[0] for q in test_queries],
                    [q[1] for q in test_queries]
                )
                st.metric("Document Hit Rate (Top-5)", f"{hit_rate*100:.1f}%")

            with st.expander("Quality", expanded=False):
                ref_answer = "The 2023 revenue was $5.2 million according to page 12."
                test_response = llm.generate_response(
                    "What was the 2023 revenue?",
                    retriever.retrieve("What was the 2023 revenue?")
                )
                scores = evaluator.evaluate_response(test_response, ref_answer)
                st.write(f"""
                - **BLEU-4 Score**: {scores['bleu-4']:.2f}
                - **ROUGE-L F1**: {scores['rouge-l']:.2f}
                """)

            with st.expander("Performance Benchmarks", expanded=False):
                latency = evaluator.measure_latency("What was 2023 revenue?")
                st.write(f"""
                - **Average Response Time**: {latency:.2f}s
                - **Embedding Speed**: {len(text_embeddings)/latency:.0f} docs/s
                """)

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")

                
                

if __name__ == "__main__":
    main()