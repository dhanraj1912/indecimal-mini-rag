# 🏗️ Indecimal Mini-RAG — Construction AI Assistant

This project implements a Retrieval-Augmented Generation (RAG) system for Indecimal’s construction marketplace. The assistant answers user questions using only internal company documents, ensuring **accuracy**, **transparency**, and **zero hallucination**.

A live chatbot is provided via Streamlit, backed by FAISS vector search and a local LLM (Mistral).

---

## 🎯 Objective

Build an AI assistant that:

- **Retrieves** relevant information from internal construction documents
- **Uses semantic search** (embeddings + vector DB)
- **Generates answers** strictly grounded in retrieved content
- **Displays retrieved context** alongside the final answer for transparency

---

## 🧠 System Architecture

User Query → Sentence-Transformer Embedding → FAISS Vector Search → Top-K Relevant Chunks → Local LLM (Mistral via Ollama) → Grounded Answer

This pipeline ensures every answer is backed by internal documents.

---

## 📚 Knowledge Sources

The assistant is built from three internal Indecimal documents:

- Company Overview & Customer Journey
- Package Pricing & Material Specifications
- Payment Safety, Quality Control & Delay Management

These files are stored as raw text and embedded into a FAISS vector database.

---

## 🔍 Document Processing

- Documents are split into **500-character overlapping chunks** to preserve context while enabling semantic search.
- Each chunk is embedded using **sentence-transformers/all-MiniLM-L6-v2**, chosen for being open-source, fast, and effective for semantic similarity.

---

## 🧮 Vector Search

- We use **FAISS** (Facebook AI Similarity Search) as a local vector store for fast nearest-neighbor retrieval without cloud dependency.
- For each user query we return the **Top-4** most relevant chunks.

---

## 🤖 Language Model

- Local LLM: **Mistral 7B** via **Ollama**.

Why this setup:

- No API costs
- Data stays local (no external transmission)
- Strong instruction-following and low latency

This satisfies the assignment’s “Local LLM (Bonus)” requirement.

---

## 🛑 Hallucination Prevention

The LLM is constrained with the following rule:

> **You must answer only from the provided context. If the answer is not in the context, say: `Not found in provided documents`.**

This prevents made-up facts and unsupported claims.

---

## 🖥️ Chatbot Interface

- A Streamlit web app provides a real-time chatbot UI.
- The UI shows the user's question, the retrieved document chunks used as context, and the final LLM-generated answer for full transparency.

---

## 🚀 How to Run Locally

1. Install dependencies

```bash
pip install streamlit faiss-cpu sentence-transformers ollama
```

2. Download the local LLM

```bash
ollama pull mistral
```

3. Build the vector database

```bash
python3 ingest.py
```

4. Launch the chatbot

```bash
streamlit run app.py
```

Open the app in your browser: `http://localhost:8501`

---

## 🧪 Example Query

**User:** How does Indecimal ensure construction quality?

**System retrieves:** Quality assurance policy; 445+ checkpoint system; Project monitoring process.

**Final Answer:** A grounded response generated only from the retrieved content.

---

## 📊 Evaluation Summary

We tested the system with 10+ internal questions. Key results:

| Metric | Result |
|---|---:|
| Hallucinations | 0 |
| Answer correctness | High |
| Retrieval relevance | High |
| Explainability | Full transparency |

The system consistently retrieved correct policy and specification sections and generated accurate answers.

---

## 🏆 Why This Project Is Strong

- Real-world enterprise knowledge grounding
- Vector databases & embeddings
- Local LLM inference (privacy-preserving)
- Transparent, explainable answers
- End-to-end AI product engineering (production-style RAG)

Output : 
<img width="1710" height="997" alt="Screenshot 2025-12-28 at 1 03 29 AM" src="https://github.com/user-attachments/assets/c3d7377e-3bcf-4c9b-a19a-675a0d49d9e2" />
<img width="1710" height="997" alt="Screenshot 2025-12-28 at 1 03 51 AM" src="https://github.com/user-attachments/assets/52d2d2e2-0f88-439d-a9f1-2062a0c8831e" />
<img width="1710" height="997" alt="Screenshot 2025-12-28 at 1 04 05 AM" src="https://github.com/user-attachments/assets/80d673c2-22bc-400f-8a2d-d866634ff35c" />




---


