# File: main.py

# --- Import necessary tools ---
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the LlamaIndex tools for RAG
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini as GeminiLLM # Renamed to avoid confusion
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Initial Setup ---

# Load the secret key from our .env file
load_dotenv()

# Tell LlamaIndex which AI models to use
# We're using Gemini for both understanding language (LLM) and for embeddings.
Settings.llm = GeminiLLM(model_name="models/gemini-flash-latest")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Create our FastAPI web application
app = FastAPI(title="DocSense Q&A API")

# A simple "database" in memory to hold our processed document.
# `index` will store the knowledge extracted from the PDF.
index = None

# --- Define the data format for our API ---

# This class defines that when a user asks a question,
# the data must be a JSON object like: {"question": "some text"}
class QueryRequest(BaseModel):
    question: str

# --- Create the API Endpoints (The Doors to Our App) ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    This endpoint handles uploading a PDF file.
    It reads the PDF, converts it into a format the AI can understand (an "index"),
    and stores it in memory.
    """
    global index
    data_path = "data"

    # Clean up any old files from previous uploads
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    file_path = os.path.join(data_path, file.filename)

    # Save the uploaded file to the "data" folder
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Now, let LlamaIndex read the PDF from that folder
    try:
        documents = SimpleDirectoryReader(data_path).load_data()
        
        # Create the index (the AI's knowledge base) from the document
        index = VectorStoreIndex.from_documents(documents)
        
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process file: {str(e)}"}

@app.post("/query")
async def query_document(request: QueryRequest):
    """
    This endpoint takes a user's question and uses the indexed document
    to find the answer with Gemini.
    """
    global index

    if index is None:
        return {"status": "error", "message": "No document is indexed. Please upload a file first."}

    # Create a "query engine," which is like a smart search tool for our index
    query_engine = index.as_query_engine()
    
    # Ask the query engine the user's question
    response = query_engine.query(request.question)
    
    # Return the answer
    return {"status": "success", "answer": str(response)}