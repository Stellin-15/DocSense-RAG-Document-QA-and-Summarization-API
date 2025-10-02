# File: main.py

import os
import shutil
import asyncio
import time # <-- Add this import to time the process
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini as GeminiLLM
from llama_index.embeddings.gemini import GeminiEmbedding

load_dotenv()
Settings.llm = GeminiLLM(model_name="models/gemini-flash-latest")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

app = FastAPI(title="DocSense Q&A API")
index = None

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global index
    data_path = "data"

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    file_path = os.path.join(data_path, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        def load_and_index():
            # --- DEBUG PRINTS ---
            print("--- Starting the indexing process in a background thread... ---")
            start_time = time.time()

            print("Step 1: Loading documents from the directory...")
            documents = SimpleDirectoryReader(data_path).load_data()
            print(f"Step 1 COMPLETE. Found {len(documents)} document(s).")

            print("Step 2: Creating the vector store index. This is the slow part (API calls)...")
            instance_index = VectorStoreIndex.from_documents(documents)
            print("Step 2 COMPLETE. Index has been created.")

            end_time = time.time()
            print(f"--- Indexing finished in {end_time - start_time:.2f} seconds. ---")
            return instance_index

        index = await asyncio.to_thread(load_and_index)
        
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed."}
    except Exception as e:
        # --- IMPORTANT DEBUG PRINT FOR ERRORS ---
        print(f"!!!!!!!! AN ERROR OCCURRED: {e} !!!!!!!!")
        return {"status": "error", "message": f"Failed to process file: {str(e)}"}

# (The /query endpoint remains the same for now)
@app.post("/query")
async def query_document(request: QueryRequest):
    global index

    if index is None:
        return {"status": "error", "message": "No document is indexed. Please upload a file first."}

    def run_query():
        query_engine = index.as_query_engine()
        return query_engine.query(request.question)

    response = await asyncio.to_thread(run_query)
    
    return {"status": "success", "answer": str(response)}