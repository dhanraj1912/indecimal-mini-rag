import pickle
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
for file in ["data/doc1.txt", "data/doc2.txt", "data/doc3.txt"]:
    with open(file, "r", encoding="utf-8") as f:
        docs.append(f.read())

chunks = []
for doc in docs:
    for i in range(0, len(doc), 500):
        chunks.append(doc[i:i+500])

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(384)
index.add(embeddings)

faiss.write_index(index, "vector_store/index.faiss")
pickle.dump(chunks, open("vector_store/chunks.pkl", "wb"))

print("Vector DB created successfully")
