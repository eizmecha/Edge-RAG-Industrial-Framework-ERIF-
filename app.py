import os
import sys
import time
import logging
import streamlit as st
import streamlit.components.v1 as components

# Local Edge AI Components
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Orchestration Logic
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Logging & Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory where the local FAISS vector database is stored
DB_FAISS_PATH = 'vectorstore/db_faiss'

# ==========================================
# 2. Local Knowledge Base Engine (RAG)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_local_system() -> tuple:
    """
    Initializes local Ollama SLM and connects the FAISS retriever.
    Cached to prevent reloading the tensor weights on every UI interaction.
    """
    try:
        # The Generative Brain: 100% Local Phi-3.5 (SLM) via Ollama
        llm = OllamaLLM(model="phi3.5", temperature=0.0)
        
        # The Retrieval Memory: Local FAISS Database mapped to HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Configure retriever to fetch the top 10 most relevant engineering chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        return llm, retriever
    except Exception as e:
        logger.error(f"Hardware/Initialization Error: {str(e)}")
        st.error(f"System Offline: Could not load local database. Ensure ingest_data.py has been run. Error: {str(e)}")
        st.stop()

# ==========================================
# 3. HMI Custom CSS Styling
# ==========================================
def inject_custom_css():
    """
    Injects custom CSS to transform the default Streamlit UI into a 
    professional Industrial SCADA dashboard.
    """
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #00A3E0; font-family: 'Segoe UI', sans-serif; }
        
        /* E-STOP Button Styling */
        div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #FF4B4B;
            color: white;
            border: 2px solid #CC0000;
            font-weight: bold;
            border-radius: 4px;
            width: 100%;
        }
        div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background-color: #CC0000;
            border-color: #990000;
        }

        /* Standard Button & Metrics */
        .stButton > button { width: 100%; border-radius: 4px; font-weight: 600; }
        div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #00FF00; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. Main HMI (User Interface)
# ==========================================
def main():
    # Page configuration must be the absolute first Streamlit command
    st.set_page_config(page_title="ERIF Terminal", page_icon="🏭", layout="wide")
    inject_custom_css()

    # --- SIDEBAR: Control Panel ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Siemens-logo.svg/1024px-Siemens-logo.svg.png", width=150)
        st.markdown("### 🎛️ ERIF Control Panel")
        st.divider()
        
        st.metric(label="Compute Engine", value="RTX 4090 Active")
        st.metric(label="Inference Model", value="Phi-3.5 (Local)")
        st.metric(label="Knowledge Base", value="FAISS Vector DB")
        st.metric(label="Network Status", value="Air-Gapped (Secure)")
        st.divider()

        # Operational Buttons
        if st.button("🧹 Clear Terminal Log"):
            st.session_state.messages = []
            st.rerun()
            
        # ==========================================
        # UPGRADED: Hardware-Level E-STOP Routine
        # ==========================================
        if st.button("🛑 EMERGENCY STOP", type="primary"):
            logger.critical("E-STOP TRIGGERED BY OPERATOR. INITIATING DOM BLACKOUT AND KERNEL TERMINATION.")
            
            # Inject Javascript to blackout the frontend and attempt to close the browser tab
            components.html(
                """
                <script>
                // 1. Blackout the entire DOM to simulate physical power loss
                window.parent.document.body.innerHTML = `
                    <div style="background-color:#000000; position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:99999; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                        <h1 style="color:#FF0000; font-family:'Courier New', monospace; font-size:5vw; font-weight:bold; margin:0;">🚨 SYSTEM HALTED</h1>
                        <p style="color:#FFFFFF; font-family:'Courier New', monospace; font-size:1.5vw;">TENSOR CORES AND BACKEND SERVER TERMINATED</p>
                    </div>
                `;
                // 2. Attempt to explicitly close the window
                setTimeout(function(){ window.close(); }, 500);
                </script>
                """,
                height=0, width=0
            )
            
            # Allow network packets containing the JS payload to reach the browser before killing Python
            time.sleep(1.5)
            
            # Force kill the backend processes safely at the OS level
            os._exit(0)

    # --- MAIN PANEL: Diagnostic Terminal ---
    st.title("🏭 Edge-RAG Industrial Framework (ERIF)")
    st.markdown("*Autonomous S7-1200 Logic & Hardware Copilot*")
    st.divider()

    # Initialize Backend Pipeline
    llm, retriever = initialize_local_system()

    # Define strict, dynamic industrial prompts
    system_prompt = (
        "You are ERIF, a Senior Industrial Automation Copilot specializing in Siemens S7-1200 PLCs.\n"
        "CRITICAL RULES:\n"
        "1. CASUAL CHAT: If the user simply says hello, greets you, or asks how you are, respond politely like a normal assistant and ask how you can help with their PLC. Do NOT provide technical steps for greetings.\n"
        "2. CONTEXT ALIGNMENT: For technical questions, read the provided Context carefully. If the Context is unrelated to the question, or if the answer is missing, you MUST reply EXACTLY with: 'Insufficient data in local memory. Please consult a senior engineer.'\n"
        "3. NO HALLUCINATIONS: Do NOT invent diagnostic steps. Do NOT reference page numbers or manuals unless they are explicitly written in the provided Context text.\n"
        "4. FORMATTING: Use bullet points for lists and procedures, but do NOT force simple definitions into a step-by-step diagnostic format.\n\n"
        "Context: {context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Enter PLC Error Code or Maintenance Query (e.g., 'How to tune PID_Compact?')..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Accessing local FAISS memory and engaging RTX Tensor Cores..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Retrieved Documentation (Transparency Audit)"):
                        for idx, doc in enumerate(response["context"]):
                            st.code(f"--- Document Chunk {idx + 1} ---\n{doc.page_content}", language="text")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    logger.error(f"Inference error: {str(e)}")
                    error_msg = f"Diagnostics Failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()

