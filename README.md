# 🏭 Edge-RAG Industrial Framework (ERIF)
### Autonomous S7-1200 Logic & Hardware Copilot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/HMI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Edge_AI-Ollama%20%7C%20Phi--3.5-black?logo=linux&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-00A3E0?logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 System Overview
The **Edge-RAG Industrial Framework (ERIF)** is a production-grade, 100% air-gapped AI Copilot designed specifically for Industrial Control Systems (ICS) and SCADA environments. ERIF utilizes Retrieval-Augmented Generation (RAG) to process thousands of pages of complex Siemens S7-1200 engineering manuals, providing automation engineers with deterministic, zero-hallucination diagnostic intelligence directly at the edge.

By running locally via Ollama and utilizing GPU-accelerated vector mathematics, ERIF guarantees strict data privacy, making it suitable for deployment in secure, isolated plant network architectures (Purdue Model Levels 2/3).

---

## ⚙️ Architectural Breakdown (Mechatronics Framework)

ERIF is engineered utilizing a multi-disciplinary system architecture:

### 1. Software Components (The Orchestration Engine)
* **HMI / SCADA Interface:** Built with `Streamlit`, featuring custom CSS to mimic industrial control panels, complete with real-time metrics and a functional E-STOP routine.
* **Data Ingestion Pipeline:** Utilizes `PDFPlumber` for high-fidelity extraction of complex structural engineering tables, memory maps, and fault codes from industrial PDFs.
* **Generative Core:** Employs the `Phi-3.5` Small Language Model (SLM) operating at `Temperature = 0.0` to enforce strict deterministic logic.

### 2. Electrical & Compute (The Hardware Acceleration)
* **Local Tensor Processing:** Vector embeddings (`all-MiniLM-L6-v2`) are generated utilizing CUDA cores, drastically reducing processing latency for 10,000+ page datasets.
* **Air-Gapped Operation:** Zero dependency on cloud APIs. All inference and retrieval operations execute entirely on local silicon.

### 3. Mechanical & Operations (Safety & Reliability)
* **E-STOP Integration:** Hardware-level termination logic embedded directly into the HMI to instantly kill background server processes during critical anomalies.
* **Audit Trail:** Built-in transparency expanders allow the operator to trace every AI-generated response back to the specific chunk of the original Siemens manual.

---

## 📂 Project Structure

```text
Industrial-SCADA-RAG/
├── app.py                  # Main Streamlit SCADA HMI & RAG Orchestration
├── ingest_data.py          # Data ingestion, PDF parsing, & vector pipeline
├── requirements.txt        # Strict Python environment dependencies
├── README.md               # System documentation & deployment guide
├── .gitignore              # Filters out large DBs and proprietary PDFs
├── data/                   # [DO NOT UPLOAD] Siemens PDFs directory
│   ├── S7-1200_System_Manual_V4.6.pdf
│   ├── S7-1200_1500_PID_Control.pdf
│   └── ... (5 additional manuals)
└── vectorstore/            # [DO NOT UPLOAD] Generated FAISS matrix
    └── db_faiss/
        ├── index.faiss     # The compiled mathematical vector space
        └── index.pkl       # Serialized metadata mapping

```
## 📚 Industrial Data Acquisition (Knowledge Base)
> **⚠️ LEGAL & IP COMPLIANCE:** Siemens proprietary engineering manuals are **NOT** included in this repository to comply with Intellectual Property and Copyright regulations.
> 
To build the ERIF Knowledge Base, you must acquire the official PDFs directly from the Siemens Industry Online Support (SIOS) portal and place them into the data/ directory.
| Engineering Discipline | Official Siemens Manual | SIOS Entry ID |
|---|---|---|
| **Core Hardware** | S7-1200 Programmable Controller System Manual | 109815048 |
| **Process Control** | S7-1200, S7-1500 PID Control Function Manual | 109741567 |
| **SCADA & Comms** | S7-1200/S7-1500 Communication Function | 109736275 |
| **SCADA & Comms** | S7-1200 Web Server Function Manual | 109773506 |
| **Motion Control** | S7-1200 Motion Control (V6.0/7.0) | 109754206 |
| **Fail-Safe Control** | SIMATIC Safety - Configuring and Programming | 54110126 |
| **Cybersecurity** | Operational Guidelines for Industrial Security | 109756589 |
## 🤖 System Capabilities & Example Queries
Once the system is compiled, ERIF functions as a Senior Automation Engineer. Here are several queries you can execute in the terminal, categorized by task, to test its local edge capabilities:
**🔴 Basic Troubleshooting & Error Codes**
 * "What does Error 16#80C4 mean?"
 * "How do I resolve a 'Module not configured' error?"
 * "What is the meaning of diagnostic code 16#80A1?"
 * "My S7-1200 shows a red ERROR LED. What should I check first?"
**⚙️ Configuration & Setup**
 * "How to configure PID_Compact?"
 * "What are the steps to set up an S7-1200 as a Modbus TCP server?"
 * "Explain how to configure an analog input for 4-20mA."
 * "How do I set the IP address of an S7-1200 using TIA Portal?"
**💻 Programming & Logic**
 * "How do I use the 'NORM_X' and 'SCALE_X' instructions together?"
 * "Provide an example of a simple motor start/stop latch circuit."
 * "How do I implement a timer on-delay (TON) instruction?"
 * "Explain the exact difference between a Function (FC) and a Function Block (FB)."
**🔄 PID Control Specifics**
 * "What is the difference between PID_Compact and PID_3Step?"
 * "How do I perform auto-tuning (pretuning) on a PID_Compact controller?"
 * "What parameters do I need to adjust if my PID loop is oscillating too much?"
 * "Explain the 'Anti-windup' feature in S7-1200 PID control."
**🛡️ Safety & Hardware Diagnostics**
 * "What are the wiring requirements for a fail-safe input module?"
 * "How do I reset a safety program to its factory state?"
 * "What is the maximum cable length for an RTD sensor connected to an S7-1200?"
 * "How do I replace a faulty CPU without losing the program?"
## 🚀 Deployment Protocol
### Prerequisites
 * Python 3.10 or higher.
 * NVIDIA GPU with CUDA support (Highly Recommended for inference speed).
 * Ollama installed and running locally.
### Step 1: System Cloning & Initialization
Clone the repository and install the strict environment dependencies.
```bash
git clone [https://github.com/eizmecha/Edge-RAG-Industrial-Framework-ERIF-.git](https://github.com/eizmecha/Edge-RAG-Industrial-Framework-ERIF-.git)
cd Edge-RAG-Industrial-Framework-ERIF-
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

```
### Step 2: Download Edge AI Models
Pull the target Small Language Model into your local Ollama server.
```bash
ollama run phi3.5

```
## 🏭 Pipeline Execution
Once your data/ folder is populated with the 7 targeted PDFs, execute the build sequence.
### 1. Ingest Data & Compile Vector DB
Run the ETL pipeline to parse tables, chunk text, and generate the FAISS matrix.
*(Note: This process is highly CPU/GPU intensive and may take up to 45 minutes depending on the volume of PDFs).*
```bash
python ingest_data.py

```
### 2. Launch ERIF Terminal
Boot the Streamlit SCADA HMI.
```bash
streamlit run app.py

```
## ⚖️ Disclaimer & Liability Notice
**FOR EDUCATIONAL AND DIAGNOSTIC PURPOSES ONLY.** This AI framework is an experimental diagnostic tool. It is not a substitute for professional engineering judgment. The authors and contributors of ERIF accept **NO LIABILITY** for any hardware damage, production downtime, physical injury, or mechanical failures resulting from the application of code or logic generated by this model on live Industrial Control Systems (ICS). Always test generated PLC logic in a simulated environment (e.g., PLCSIM / Factory IO) and verify against official OEM documentation before deploying to production machinery.
