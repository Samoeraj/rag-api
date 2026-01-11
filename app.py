from fastapi import FastAPI
import chromadb
import ollama

app = FastAPI()
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")

@app.post("/query")
def query(q:str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else "No context found"
    
    answer = ollama.generate(model="tinyllama", prompt=f"Context: {context}\nQuestion: {q}\nAnswer clearly and concisely:")
    return {"answer": answer["response"]}

@app.post("/add")
def add(text:str):
    try:
        import uuid
        id=str(uuid.uuid4())
        collection.add(documents=[text], ids=[id])
        return {"message": "Text added successfully", "id": id}
    except Exception as e:
        return {"message": "Error adding text", "error": str(e)}