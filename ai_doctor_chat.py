import streamlit as st
import os
import re
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import PyPDF2
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File paths and model information
PDF_PATH = r"D:\nextastra\medical.pdf"
VECTOR_DB_PATH = r"D:\nextastra\vectordb"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------- PDF PROCESSOR ---------------- #
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path):
        # Read PDF and extract text, add page markers for reference
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        text += f"\n[PAGE {i+1}]\n{t}\n"
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
        return text

    @staticmethod
    def chunk_text(text, chunk_size=800, overlap=100):
        # Split text into overlapping chunks with page references
        chunks = []
        pages = text.split("[PAGE ")
        
        for p in pages[1:]:
            match = re.match(r'(\d+)\](.*)', p, re.DOTALL)
            if not match:
                continue
                
            page_num, content = match.groups()
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            current = []
            size = 0
            
            for s in sentences:
                words = len(s.split())
                if size + words > chunk_size and current:
                    chunk = " ".join(current)
                    chunks.append((chunk, int(page_num)))
                    # Keep overlap
                    overlap_text = " ".join(current[-3:])
                    current = [overlap_text]
                    size = len(overlap_text.split())
                current.append(s)
                size += words
                
            if current:
                chunks.append((" ".join(current), int(page_num)))
                
        return chunks


# ---------------- VECTOR DB ---------------- #
class VectorDB:
    def __init__(self):
        # Initialize vector DB paths and embedding model
        self.path = Path(VECTOR_DB_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "faiss_index.bin"
        self.chunk_file = self.path / "chunks.pkl"
        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def build(self, chunks):
        # Build FAISS index from chunks and save to disk
        self.chunks = chunks
        texts = [c[0] for c in chunks]
        
        print("Encoding chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        arr = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(arr)

        # Save to disk
        faiss.write_index(self.index, str(self.index_file))
        with open(self.chunk_file, "wb") as f:
            pickle.dump(chunks, f)
        
        print(f"Vector DB built with {len(chunks)} chunks")

    def load(self):
        # Load FAISS index and chunks from disk if they exist
        if self.index_file.exists() and self.chunk_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.chunk_file, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"Loaded vector DB with {len(self.chunks)} chunks")
            return True
        return False

    def search(self, query, k=5):
        # Search top-k most similar chunks for a query
        emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                chunk, page = self.chunks[idx]
                results.append((chunk, page))
        return results


# ---------------- RAG ENGINE ---------------- #
class MedicalRAG:
    def __init__(self, vector_db):
        self.vdb = vector_db
        self.client = Groq(api_key=GROQ_API_KEY)

    def answer(self, query, conversation_context=None):
        # Get top chunks from vector DB for context
        # Search vector DB for relevant context
        results = self.vdb.search(query, k=5)
        context = "\n\n".join([f"[Reference from Page {p}]\n{c}" for c, p in results])

        # Build enhanced prompt with conversation context
        if conversation_context:
            prompt = f"""You are an expert medical AI assistant with access to medical knowledge.

PATIENT CONVERSATION CONTEXT:
{conversation_context}

MEDICAL KNOWLEDGE BASE:
{context}

PATIENT QUESTION:
{query}

Provide a clear, medically accurate answer. Consider both the conversation context and the medical knowledge base. 
If the question relates to symptoms or conditions mentioned in the conversation, reference them appropriately.
Keep the answer professional, empathetic, and easy to understand.
"""
        else:
            prompt = f"""You are an expert medical AI assistant.

MEDICAL KNOWLEDGE BASE:
{context}

QUESTION:
{query}

Provide a clear, accurate, and medically sound answer based on the knowledge base provided.
"""

        try:
            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            answer = res.choices[0].message.content
        except Exception as e:
            answer = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            results = []

        return answer, results


# ---------------- INITIALIZE RAG SYSTEM ---------------- #
@st.cache_resource
def init_rag():
    # Initialize RAG system with caching
    try:
        vdb = VectorDB()
        
        # Try to load existing index
        if not vdb.load():
            # Build new index if not exists
            if not os.path.exists(PDF_PATH):
                st.error(f"PDF file not found at: {PDF_PATH}")
                st.info("Please update PDF_PATH in ai_doctor_chat.py")
                return None
            
            st.info("Building vector database for the first time... This may take a few minutes.")
            text = PDFProcessor.extract_text(PDF_PATH)
            if not text:
                st.error("Failed to extract text from PDF")
                return None
                
            chunks = PDFProcessor.chunk_text(text)
            if not chunks:
                st.error("Failed to create chunks from PDF")
                return None
                
            vdb.build(chunks)
            st.success("Vector database built successfully!")
        
        return MedicalRAG(vdb)
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None


# ---------------- CHAT POPUP UI ---------------- #
def ai_doctor_chat_popup(transcript, diseases, symptoms):
    # Main chat interface for medical AI
    
    # Initialize RAG system
    rag = init_rag()
    
    if rag is None:
        st.error("Failed to initialize AI Doctor. Please check your configuration.")
        return

    @st.dialog("💬 AI Doctor - Medical Knowledge Assistant", width="large")
    def chat_ui():
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Header
        st.markdown("### 🩺 AI Medical Assistant")
        st.markdown("Ask questions about your symptoms, diagnosis, or treatment options.")
        
        # Show detected symptoms and diseases
        with st.expander("📋 Detected Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Detected Symptoms:**")
                if symptoms:
                    for sym in symptoms:
                        st.markdown(f"• {sym}")
                else:
                    st.markdown("None detected")
            
            with col2:
                st.markdown("**Possible Conditions:**")
                if diseases:
                    for dis in diseases:
                        st.markdown(f"• {dis}")
                else:
                    st.markdown("None detected")
        
        st.markdown("---")
        
        # Display chat history
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)
        
        # Chat input
        user_question = st.chat_input("💭 Ask your medical question...")

        if user_question:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)

            # Prepare conversation context
            context_info = f"""
Detected Symptoms: {', '.join(symptoms) if symptoms else 'None'}
Possible Conditions: {', '.join(diseases) if diseases else 'None'}

Conversation Transcript:
{transcript[:1000]}...
"""

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing medical knowledge..."):
                    answer, sources = rag.answer(user_question, context_info)
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Medical References Used"):
                            for i, (chunk, page) in enumerate(sources, 1):
                                st.markdown(f"**Reference {i} (Page {page}):**")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.markdown("---")
            
            # Add assistant response to history
            st.session_state.chat_history.append(("assistant", answer))
        
        # Clear chat button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Open the dialog
    chat_ui()






















