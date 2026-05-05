import os
# Force strictly offline inference for air-gapped SCADA environments
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import sys
import time
import logging
import streamlit as st
import streamlit.components.v1 as components

# Local Edge AI Components
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Orchestration Logic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Enterprise Configuration & Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory where the local FAISS vector database is stored
DB_FAISS_PATH = 'vectorstore/db_faiss'

# ==========================================
# 2. Local Knowledge Base Engine (RAG Memory)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_vector_memory():
    """
    Caches the FAISS database and HuggingFace Embeddings model in strict offline mode.
    """
    try:
        # Utilize the same embedding model used during ingestion, strictly offline
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={"device": "cpu", "local_files_only": True}
        )
        
        # Load the local FAISS index
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Configure retriever: Top 5 chunks, strict 0.4 similarity threshold
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": 5, "score_threshold": 0.4}
        )
        
        return retriever
    except Exception as e:
        logger.error(f"Hardware/Initialization Error: {str(e)}")
        st.error(f"System Offline: Could not load local database. Ensure ingest_data.py has been run. Error: {str(e)}")
        st.stop()

# ==========================================
# 3. SCADA HMI Custom CSS Styling
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #00A3E0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #FF4B4B; color: white; border: 2px solid #CC0000; font-weight: bold; border-radius: 4px; width: 100%;
        }
        div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background-color: #CC0000; border-color: #990000;
        }
        
        .stButton > button { width: 100%; border-radius: 4px; font-weight: 600; }
        div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #00FF00; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. Main HMI (User Interface & Logic)
# ==========================================
def main():
    st.set_page_config(page_title="ERIF Terminal", page_icon="🏭", layout="wide")
    inject_custom_css()

    # --- SIDEBAR: Control Panel ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Siemens-logo.svg/1024px-Siemens-logo.svg.png", width=150)
        st.markdown("### 🎛️ ERIF Control Panel")
        st.divider()
        
        st.metric(label="Compute Engine", value="RTX 4090 / CPU Active")
        st.metric(label="Inference Model", value="Phi-3.5 (Local)")
        st.metric(label="Knowledge Base", value="FAISS Vector DB")
        st.metric(label="Network Status", value="Air-Gapped (Secure)")
        st.divider()

        if st.button("🧹 Clear Terminal Log"):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🛑STOP/EXIT", type="primary"):
            logger.critical("E-STOP TRIGGERED BY OPERATOR. INITIATING DOM BLACKOUT AND KERNEL TERMINATION.")
            components.html(
                """
                <script>
                window.parent.document.body.innerHTML = `
                    <div style="background-color:#000000; position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                        <h1 style="color:#FF0000; font-family:'Courier New', monospace; font-size:5vw; font-weight:bold; margin:0;">🚨 SYSTEM HALTED</h1>
                        <p style="color:#FFFFFF; font-family:'Courier New', monospace; font-size:1.5vw;">TENSOR CORES AND BACKEND SERVER TERMINATED</p>
                    </div>
                `;
                setTimeout(function(){ window.close(); }, 500);
                </script>
                """, height=0, width=0
            )
            time.sleep(1.5)
            os._exit(0)

    # --- MAIN PANEL ---
    st.title("🏭 Edge-RAG Industrial Framework (ERIF)")
    st.markdown("*Autonomous S7-1200 Logic & Hardware Copilot*")
    st.divider()

    # ==========================================
    # 5. ENTERPRISE ORCHESTRATION (Two-Layer Defense)
    # ==========================================
    llm = OllamaLLM(model="phi3.5", temperature=0.0)
    retriever = initialize_vector_memory()

    # Layer 1: Strict XML-Based System Prompt
    system_prompt = (
        "You are ERIF, a strict Senior Industrial Automation Copilot for Siemens S7-1200 PLCs.\n"
        "You operate in a highly secure, air-gapped environment. Obey these rules absolutely:\n\n"
        "<RULES>\n"
        "1. CASUAL GREETINGS: If the user simply says hello, respond politely and ask how you can help with their PLC.\n"
        "2. STRICT RELIANCE: For technical questions, you must base your answer EXCLUSIVELY on the provided <CONTEXT>.\n"
        "3. NO HALLUCINATION: Do NOT invent diagnostic steps. Do NOT reference page numbers unless explicitly written in the <CONTEXT>.\n"
        "4. OUT OF DOMAIN: If the <CONTEXT> is empty, or if it does NOT contain the exact answer to the user's prompt (e.g., recipes, general health, non-Siemens topics), you are FORBIDDEN from answering from internal knowledge.\n"
        "5. FALLBACK STRING: If forbidden by Rule 4, you MUST reply with this exact phrase and nothing else: 'Insufficient data in local memory. Please consult a senior engineer.'\n"
        "When answering about sequential processes or multiple modes (e.g., Pretuning vs Fine Tuning), explicitly separate their conditions and do not mix them."
        "</RULES>\n\n"
        "<CONTEXT>\n"
        "{context}\n"
        "</CONTEXT>"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # We use create_stuff_documents_chain directly to maintain explicit control over the context
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Enter PLC Error Code or Maintenance Query..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Accessing local FAISS memory and engaging Inference Engine..."):
                try:
                    # Layer 2: Python-Level Guardrail (The Interceptor)
                    # Manually fetch documents to evaluate them BEFORE waking up the LLM
                    retrieved_docs = retriever.invoke(user_query)

                    # Detect simple greetings to bypass the strict context drop
                    casual_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
                    is_greeting = user_query.strip().lower() in casual_greetings

                    # SCENARIO 1: It's a simple greeting -> Python answers instantly
                    if is_greeting:
                        answer = "Hello! I am ERIF, your Senior Industrial Automation Copilot. How can I assist you with your Siemens S7-1200 PLC today?"
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    # SCENARIO 2: No context found -> Python blocks it instantly
                    elif not retrieved_docs:
                        answer = "Insufficient data in local memory. Please consult a senior engineer."
                        st.markdown(answer)
                        with st.expander("🔍 View Retrieved Documentation (Transparency Audit)"):
                            st.warning("No relevant Siemens S7-1200 engineering context found for this query. LLM execution bypassed for safety.")
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    # SCENARIO 3: Context found -> Pass to the LLM
                    else:
                        answer = document_chain.invoke({
                            "input": user_query,
                            "context": retrieved_docs
                        })
                        st.markdown(answer)
                        if retrieved_docs:
                            with st.expander("🔍 View Retrieved Documentation (Transparency Audit)"):
                                for idx, doc in enumerate(retrieved_docs):
                                    st.code(f"--- Document Chunk {idx + 1} ---\n{doc.page_content}", language="text")
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                except Exception as e:
                    logger.error(f"Inference error: {str(e)}")
                    error_msg = f"Diagnostics Failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
