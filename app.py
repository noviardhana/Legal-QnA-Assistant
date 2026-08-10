"""
Legal Q&A Assistant - Streamlit Dashboard

A Retrieval-Augmented Generation (RAG) chatbot powered by:
- 4 Indonesian legal documents (Undang-Undang)
- Fine-tuned LLaMA3 model with GRPO training
- Hybrid retrieval (semantic + BM25 ensemble)
- Cross-encoder reranking for relevance

The application uses HuggingFace Inference API for lightweight inference 
(no GPU needed on Streamlit Cloud).

Repository structure required:
  app.py
  index/
    faiss_index/          <- FAISS vector store (created by build_index.py)
    child_docs.pkl        <- Small document chunks
    parent_docs.pkl       <- Large context documents
  requirements.txt
  .streamlit/secrets.toml (local) or Secrets in Streamlit Cloud:
    HF_MODEL_REPO = "username-hf/llama3-legal-id-grpo"
    HF_API_TOKEN  = "hf_xxxxxxxxxxxxxxxxxxxx"
"""

import os
import pickle
from datetime import datetime

import requests
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

try:
    from langchain.retrievers import EnsembleRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers import EnsembleRetriever

# =========================================================
# Configuration
# =========================================================
st.set_page_config(
    page_title="Legal Q&A Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Index directory paths
INDEX_DIR = "index"
FAISS_DIR = os.path.join(INDEX_DIR, "faiss_index")

# Retrieval hyperparameters
ENSEMBLE_VECTOR_WEIGHT = 0.6       # Weight for semantic search in ensemble
ENSEMBLE_BM25_WEIGHT = 0.4         # Weight for BM25 in ensemble
RETRIEVAL_TOP_K = 5                # Number of docs to retrieve before reranking
RERANK_TOP_K = 3                   # Number of docs to rerank (for context)
RELEVANCE_THRESHOLD = 0.0           # Minimum rerank score (tune based on your data)

# HuggingFace Model Configuration
HF_MODEL_REPO = st.secrets.get("HF_MODEL_REPO", os.environ.get("HF_MODEL_REPO", ""))
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.environ.get("HF_API_TOKEN", ""))

# Models for embedding and reranking
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RAG prompt template (in Indonesian for better legal context)
RAG_PROMPT_TEMPLATE = """You are a legal assistant specializing in Indonesian law.
Answer the question ONLY based on the provided legal context.
Include source citations at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""


# =========================================================
# Resource Loading (Cached per Streamlit session)
# =========================================================
@st.cache_resource(show_spinner="Loading document index...")
def load_retriever():
    """
    Load FAISS vector store and create hybrid retriever (semantic + BM25).
    
    This function is cached to avoid reloading on every interaction.
    Combines:
    - Semantic retrieval (dense embeddings for meaning)
    - BM25 retrieval (sparse retrieval for exact keywords)
    
    Returns:
        tuple: (ensemble_retriever, parent_docs_dict)
               - ensemble_retriever: Hybrid retriever combining both methods
               - parent_docs: Dict mapping parent_id -> Document (for context)
    
    Raises:
        FileNotFoundError: If index folder or pickle files don't exist
        RuntimeError: If embedding model fails to load
    """
    # Load embedding model and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    
    # Load pickled documents
    with open(os.path.join(INDEX_DIR, "child_docs.pkl"), "rb") as f:
        child_docs = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "parent_docs.pkl"), "rb") as f:
        parent_docs = pickle.load(f)
    
    # Create semantic retriever (vector similarity search)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    
    # Create BM25 retriever (keyword/term frequency search)
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = RETRIEVAL_TOP_K
    
    # Combine both retrievers with ensemble weights
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[ENSEMBLE_VECTOR_WEIGHT, ENSEMBLE_BM25_WEIGHT]
    )
    
    return ensemble, parent_docs


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker():
    """
    Load cross-encoder model for reranking retrieved documents.
    
    Cross-encoders score document relevance relative to the query.
    This provides more accurate ranking than vector similarity alone.
    
    Returns:
        CrossEncoder: Loaded cross-encoder model for scoring
    
    Raises:
        RuntimeError: If model fails to load from HuggingFace
    """
    return CrossEncoder(RERANK_MODEL)


# =========================================================
# RAG Pipeline Functions
# =========================================================
def retrieve_parent_chunks(query, ensemble_retriever, parent_docs):
    """
    Retrieve parent (large context) documents based on query.
    
    Flow:
    1. Use ensemble retriever to find relevant child chunks
    2. Extract parent documents (via parent_id in metadata)
    3. Return unique parents (deduplication)
    
    Args:
        query (str): User's question
        ensemble_retriever: Hybrid retriever (semantic + BM25)
        parent_docs (dict): Mapping of parent_id -> Document
    
    Returns:
        list: Unique parent Document objects (context for LLM)
    """
    # Retrieve child chunks
    child_hits = ensemble_retriever.invoke(query)
    
    # Extract unique parent documents
    seen = set()
    parents = []
    for child in child_hits:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            parents.append(parent_docs[parent_id])
    
    return parents


def rerank_documents(question, candidate_docs, reranker, top_k=RERANK_TOP_K):
    """
    Score and rerank documents using cross-encoder.
    
    Reranking improves relevance by using a task-specific scoring model
    that considers both query and document content jointly (unlike dense embeddings).
    
    Args:
        question (str): User's question for scoring context
        candidate_docs (list): Document objects to rerank
        reranker (CrossEncoder): Cross-encoder model
        top_k (int): Number of top documents to return
    
    Returns:
        tuple: (ranked_docs_with_scores, top_score)
               - ranked_docs_with_scores: [(Document, score), ...] sorted by relevance
               - top_score: Highest relevance score (for threshold checking)
    
    Example:
        >>> ranked, best_score = rerank_documents(q, docs, reranker)
        >>> if best_score < THRESHOLD:
        ...     # No relevant docs found, use fallback
    """
    # Create query-document pairs for scoring
    pairs = [(question, doc.page_content) for doc in candidate_docs]
    
    # Score all pairs
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    # Sort by score descending
    ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Get top-k results
    top_ranked = ranked[:top_k]
    top_score = float(top_ranked[0][1]) if top_ranked else float("-inf")
    
    return top_ranked, top_score


def web_fallback(question, max_results=3):
    """
    Fallback search using DuckDuckGo web search.
    
    Used when:
    - No relevant documents found in legal index
    - Relevance score below threshold
    
    Args:
        question (str): Query to search
        max_results (int): Number of results to return
    
    Returns:
        list: List of formatted web search results, or empty list if failed
    """
    try:
        from ddgs import DDGS
        results = DDGS().text(question, max_results=max_results)
        return [
            f"[web: {r.get('href', '')}] {r.get('title', '')} - {r.get('body', '')}"
            for r in results
        ]
    except Exception as e:
        return []


def get_context(question, ensemble_retriever, parent_docs, reranker):
    """
    Main context retrieval pipeline.
    
    Flow:
    1. Retrieve parent chunks using ensemble retriever
    2. Rerank using cross-encoder
    3. Check relevance threshold
    4. Use web fallback if needed
    
    Args:
        question (str): User's question
        ensemble_retriever: Hybrid retriever
        parent_docs (dict): Parent document mapping
        reranker (CrossEncoder): Reranking model
    
    Returns:
        tuple: (context_texts, citations)
               - context_texts: List of formatted context strings for LLM
               - citations: List of source citations for user feedback
    """
    # Step 1: Retrieve parent chunks
    parents = retrieve_parent_chunks(question, ensemble_retriever, parent_docs)
    if not parents:
        return web_fallback(question), []
    
    # Step 2: Rerank documents
    top_ranked, top_score = rerank_documents(question, parents, reranker)
    
    # Step 3: Check relevance threshold
    if top_score < RELEVANCE_THRESHOLD:
        return web_fallback(question), []
    
    # Step 4: Format context and citations
    context_texts = []
    citations = []
    
    for doc, score in top_ranked:
        meta = doc.metadata
        # Format citation
        citation = (
            f"{meta.get('source')} "
            f"(Law No. {meta.get('law_number')}/{meta.get('law_year')})"
        )
        citations.append(citation)
        
        # Format context for LLM
        context = f"[{meta.get('source')}] {doc.page_content}"
        context_texts.append(context)
    
    return context_texts, citations


def call_hf_inference(prompt, max_new_tokens=400):
    """
    Call HuggingFace Inference API to generate answer.
    
    Uses the fine-tuned GRPO model deployed on HF Inference API.
    Configuration:
    - Model: llama3-legal-id-grpo (fine-tuned for Indonesian law)
    - Temperature: 0.3 (low for legal precision)
    - Max tokens: 400 (concise legal answers)
    
    Args:
        prompt (str): RAG prompt with context and question
        max_new_tokens (int): Maximum tokens to generate
    
    Returns:
        str: Generated answer from model, or error message if failed
    
    Raises:
        requests.RequestException: API connection errors (handled gracefully)
    """
    if not HF_MODEL_REPO or not HF_API_TOKEN:
        return (
            "⚠️ HuggingFace configuration missing. "
            "Set HF_MODEL_REPO and HF_API_TOKEN in Streamlit Secrets."
        )
    
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.3,        # Low temperature for precision
            "return_full_text": False,  # Only return new tokens
        },
    }
    
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle different response formats
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        
        if isinstance(data, dict) and "error" in data:
            return f"⚠️ Model loading on HF Inference API. Try again in a moment: {data['error']}"
        
        return str(data)
    
    except requests.exceptions.RequestException as e:
        return f"⚠️ Failed to reach HuggingFace Inference API: {e}"


def rag_answer(question, ensemble_retriever, parent_docs, reranker):
    """
    Complete RAG pipeline: retrieve, rerank, generate, cite.
    
    End-to-end flow:
    1. Get relevant context via get_context()
    2. Format RAG prompt with context and question
    3. Call GRPO model for generation
    4. Append source citations
    
    Args:
        question (str): User's legal question
        ensemble_retriever: Hybrid retriever
        parent_docs (dict): Parent document mapping
        reranker (CrossEncoder): Reranking model
    
    Returns:
        str: Generated answer with citations
    """
    # Retrieve and rerank documents
    context_texts, citations = get_context(
        question, ensemble_retriever, parent_docs, reranker
    )
    
    # Format context for LLM
    context = "\n\n".join(context_texts)
    
    # Fill RAG prompt template
    filled_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Generate answer
    answer = call_hf_inference(filled_prompt)
    
    # Append citations
    if citations:
        answer += "\n\n**Sources:** " + "; ".join(sorted(set(citations)))
    
    return answer


# =========================================================
# Streamlit UI
# =========================================================
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .main .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Legal Q&A Assistant")
st.caption(
    "Retrieval-Augmented Generation chatbot powered by fine-tuned LLaMA3 (GRPO training) "
    "and 4 Indonesian legal documents. Ask about employment law, overtime, leave, and worker rights."
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Chat controls and examples
with st.sidebar:
    st.header("💬 Chat History")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.session_state.chat_history:
            chat_text = "\n\n".join(
                f"[{m['time']}] {m['role'].upper()}: {m['content']}"
                for m in st.session_state.chat_history
            )
            st.download_button(
                "⬇️ Download",
                chat_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True,
            )
    
    st.divider()
    st.subheader("Example Questions")
    example_questions = [
        "I worked overtime 3 hours yesterday. Am I entitled to overtime pay?",
        "How many days of annual leave must the company provide?",
        "What are worker rights when terminated without cause?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
            st.session_state["pending_question"] = q
    
    st.divider()
    st.caption(f"Total messages: {len(st.session_state.chat_history)}")

# Load resources (cached by Streamlit)
try:
    ensemble_retriever, parent_docs = load_retriever()
    reranker = load_reranker()
    resources_ready = True
except Exception as e:
    resources_ready = False
    st.error(
        f"Failed to load document index. Ensure 'index/' folder exists in repository "
        f"(run build_index.py first). Error: {e}"
    )

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and processing
question = st.chat_input("Ask your legal question...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question and resources_ready:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            answer = rag_answer(question, ensemble_retriever, parent_docs, reranker)
        st.markdown(answer)
    
    # Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%H:%M:%S")
    })

elif question and not resources_ready:
    st.warning("Document index not ready — see error message above.")