import streamlit as st
import faiss
import pickle
import requests
import os
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector DB
index = faiss.read_index("vector_store/index.faiss")
chunks = pickle.load(open("vector_store/chunks.pkl", "rb"))

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_llm(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    return res.json()["choices"][0]["message"]["content"]

st.title("🏗 Indecimal AI Assistant")

query = st.text_input("Ask about construction, pricing, quality, delays, etc")

if query:
    q_emb = model.encode([query])
    D, I = index.search(q_emb, k=4)

    retrieved = [chunks[i] for i in I[0]]

    st.subheader("📄 Retrieved Context")
    for c in retrieved:
        st.write(c)

    prompt = f"""
You must answer ONLY from the context below.
If the answer is not in the context, say "Not found in provided documents."

Context:
{retrieved}

Question: {query}
Answer:
"""

    answer = ask_llm(prompt)

    st.subheader("🤖 Final Answer")
    st.write(answer)
