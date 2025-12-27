import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
import ollama

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/index.faiss")
chunks = pickle.load(open("vector_store/chunks.pkl","rb"))

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
If not found, say "Not found in provided documents."

Context:
{retrieved}

Question: {query}
Answer:
"""

    response = ollama.generate(model="mistral", prompt=prompt)

    st.subheader("🤖 Final Answer")
    st.write(response["response"])
