# Local AI Agent (RAG with ChromaDB)

A custom local AI agent built with Python, LangChain, and ChromaDB for retrieval-augmented generation (RAG) over restaurant review data.

## Features
* **Vector Store Integration:** Uses ChromaDB to index structured review data (`restaurant_reviews.csv`).
* **Retrieval Pipeline:** Custom similarity search (`vector.py`) feeding context into the main agent execution loop (`main.py`).

## Setup Instructions

1. **Clone the repository:**
   ```bash

   git clone [https://github.com/YOUR_USERNAME/local-ai-agent.git](https://github.com/YOUR_USERNAME/local-ai-agent.git)
   cd local-ai-agent


2. **Create and activate a virtual environment:**
    ```bash

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install dependencies:**
    ```bash

    pip install -r requirements.txt

4. **Run the vector indexer and start the agent:**
    ```bash

    python vector.py
    python main.py