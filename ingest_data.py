import os
import logging
import glob
import time

# Local Edge AI & Parsing Components
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. Enterprise Configuration & Logging Setup
# ==========================================
# Configure standard logging to output detailed system states, timestamps, and errors
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directory where the local FAISS vector database will be persistently stored
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_local_knowledge_base():
    """
    Industrial Data Ingestion Pipeline:
    Reads Siemens S7-1200 PDF manuals, processes them using PDFPlumber to preserve 
    critical tabular data (Fault Codes, Memory Maps), generates embeddings locally 
    via HuggingFace utilizing the CPU, and indexes them in a FAISS database.
    """
    try:
        start_time = time.time()
        
        # ==========================================
        # 2. Advanced Document Intelligence (Data Ingestion)
        # ==========================================
        logger.info("Initializing ERIF Data Pipeline...")
        logger.info("Scanning for Siemens industrial manuals in the 'data/' directory...")
        
        # Locate all PDF files within the data folder
        pdf_files = glob.glob("data/*.pdf")
        
        if not pdf_files:
            logger.warning("CRITICAL HALT: No PDF documents found. Ensure manuals are inside the 'data/' folder.")
            return

        documents = []
        logger.info(f"Detected {len(pdf_files)} PDF manuals. Commencing table-aware extraction...")
        
        # Iteratively extract high-fidelity text and tables from each manual
        for i, file_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Parsing: {os.path.basename(file_path)}")
            # PDFPlumber ensures structural integrity of tables and technical diagrams
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"    -> Extracted {len(docs)} pages from {os.path.basename(file_path)}")
            
        logger.info(f"Phase 1 Complete: Successfully loaded {len(documents)} total pages.")

        # ==========================================
        # 3. NLP Text Chunking
        # ==========================================
        # Split large technical manuals into manageable logical chunks.
        # chunk_size=1000 ensures technical context is preserved.
        # chunk_overlap=200 prevents cutting engineering sentences or PLC codes in half.
        logger.info("Phase 2: Applying structural NLP chunking to the documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Phase 2 Complete: Generated {len(chunks)} semantic text chunks for vectorization.")

        # ==========================================
        # 4. CPU-Based Embeddings (Text-to-Math)
        # ==========================================
        # Explicitly routing vector mathematical operations to the CPU (Universal Compatibility)
        logger.info("Phase 3: Initializing HuggingFace Embeddings model (all-MiniLM-L6-v2) on CPU...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # CRITICAL: Forces CPU execution to bypass CUDA errors
        )

        # ==========================================
        # 5. Local FAISS Vector Database Creation
        # ==========================================
        logger.info("Phase 4: Compiling the local FAISS vector memory matrix via CPU...")
        # This operation will utilize the CPU threads to handle the 10,000+ chunks
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Ensure the target directory exists before saving
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        # Save the vector store persistently to the solid-state drive
        logger.info(f"Saving indexed vector database locally to: {DB_FAISS_PATH}")
        vector_store.save_local(DB_FAISS_PATH)
        
        execution_time = round((time.time() - start_time) / 60, 2)
        logger.info(f"SUCCESS! The High-Fidelity ERIF Knowledge Base has been built in {execution_time} minutes.")

    except Exception as e:
        logger.error(f"SYSTEM FAILURE: An unexpected hardware or software error occurred: {str(e)}")

# ==========================================
# Application Execution Point
# ==========================================
if __name__ == "__main__":
    create_local_knowledge_base()
