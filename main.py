import os
import shutil
import asyncio
import time
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import LlamaIndex and response components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini as GeminiLLM
from llama_index.embeddings.gemini import GeminiEmbedding
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --- Initial Setup ---
load_dotenv()

# Configure LlamaIndex to use Google Gemini
Settings.llm = GeminiLLM(model_name="models/gemini-flash-latest")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Create our FastAPI web application
app = FastAPI(title="DocSense Q&A API")

# Mount the 'static' directory to serve our frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory "database"
index = None

# --- Pydantic Models for API Data Structure ---

class QueryRequest(BaseModel):
    question: str

class SourceNode(BaseModel):
    file_name: str
    text: str

class StreamResponse(BaseModel):
    answer_delta: str

class FinalResponse(BaseModel):
    final_answer: str
    sources: list[SourceNode]

# --- API Endpoints ---

@app.get("/")
async def get_root():
    """Serves the main frontend HTML file."""
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document, processes it with a timeout, and creates a queryable index."""
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
            print("--- Starting indexing in background thread... ---")
            documents = SimpleDirectoryReader(data_path).load_data()
            print(f"--- Loaded {len(documents)} document(s). Creating index... ---")
            instance_index = VectorStoreIndex.from_documents(documents)
            print("--- Indexing complete. ---")
            return instance_index

        task = asyncio.to_thread(load_and_index)
        index = await asyncio.wait_for(task, timeout=90.0)
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed."}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Document processing timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/stream_query")
async def stream_query_document(request: QueryRequest):
    """Asks a question and streams the answer back token-by-token with sources."""
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="No document indexed. Please upload first.")

    async def event_stream():
        def run_streaming_query():
            chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
            return chat_engine.stream_chat(request.question)

        try:
            streaming_response = await asyncio.to_thread(run_streaming_query)
            full_answer = ""
            
            # First, stream the answer tokens
            for token in streaming_response.response_gen:
                delta = StreamResponse(answer_delta=token)
                yield f"data: {delta.json()}\n\n"
                full_answer += token
                await asyncio.sleep(0.01)

            # Then, package and stream the final response with sources
            source_nodes = []
            for node in streaming_response.source_nodes:
                # --- THIS IS THE FIX ---
                # We use os.path.basename to get just the filename from the full path
                clean_file_name = os.path.basename(node.metadata.get('file_name', 'N/A'))
                
                source_nodes.append(SourceNode(
                    file_name=clean_file_name,
                    text=node.get_content()
                ))
            
            final_response = FinalResponse(final_answer=full_answer, sources=source_nodes)
            yield f"data: {final_response.json()}\n\n"
        
        except Exception as e:
            error_response = {"error": f"An error occurred during streaming: {str(e)}"}
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")