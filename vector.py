#logic for embbeded
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import os
import pandas as pd
import requests


class OllamaDirectEmbeddings(Embeddings):
    """Calls Ollama's /api/embed endpoint directly via HTTP, bypassing
    the ollama python client (which has a bug in this environment)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _embed(self, texts):
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts):
        # Batch in chunks to avoid overly large single requests
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._embed(batch))
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} documents")
        return all_embeddings

    def embed_query(self, text):
        return self._embed([text])[0]


#df = data frame
df = pd.read_csv("restaurant_reviews.csv")
df = df.fillna("")
embeddings = OllamaDirectEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Restaurant"] + " " + row["Review"],
            metadata={"time": row["Time"]},
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

    print(f"Built {len(documents)} documents")

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

#add documents to the vector store if they don't already exist
if add_documents:
    chroma_batch_size = 5000
    for i in range(0, len(documents), chroma_batch_size):
        batch_docs = documents[i:i + chroma_batch_size]
        batch_ids = ids[i:i + chroma_batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"Added {min(i + chroma_batch_size, len(documents))}/{len(documents)} documents to vector store")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
