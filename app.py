from fastapi import FastAPI
import chromadb
import ollama
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",)

MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")
logging.info(f"Using model: {MODEL_NAME}")


app = FastAPI()
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")
ollama_client = ollama.Client(host="http://localhost:11434")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(q:str):
    logging.info(f"Querying: {q}")
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else "No context found"
    
    answer = ollama_client.generate(model=MODEL_NAME, prompt=f"Context: {context}\nQuestion: {q}\nAnswer clearly and concisely:")
    return {"answer": answer["response"]}

@app.post("/add")
def add(text:str):
    logging.info(f"Adding text: {text}")
    try:
        import uuid
        id=str(uuid.uuid4())
        collection.add(documents=[text], ids=[id])
        logging.info(f"Text added successfully: {id}")
        
        return {"message": "Text added successfully", "id": id}
    except Exception as e:
        logging.error(f"Error adding text: {e}")
        return {"message": "Error adding text", "error": str(e)}

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # Check if mock mode is enabled
    use_mock = os.getenv("USE_MOCK_LLM", "0") == "1"
    
    if use_mock:
        # Return retrieved context directly (deterministic!)
        return {"answer": context}
    else:
        # Use real LLM (production mode)
        answer = ollama.generate(
            model="tinyllama",
            prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
        )
        return {"answer": answer["response"]}



