import os
import shutil
import asyncio
import time
import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, HTTPException
from pantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.gemini import Gemini as GeminiLLM
from llama_index.embeddings.gemini import GeminiEmbedding
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --- NEW: Import the smarter PDF reader ---
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader

load_dotenv()
Settings.llm = GeminiLLM(model_name="models/gemini-flash-latest")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

app = FastAPI(title="DocSense Q&A API")
app.mount("/static", StaticFiles(directory="static"), name="static")
index = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
class ScrapeRequest(BaseModel):
    url: HttpUrl
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
    return FileResponse("static/index.html")

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
            print("--- Starting indexing with ADVANCED parser... ---")
            
            # --- THE UPGRADE IS HERE ---
            # We explicitly tell the reader to use the UnstructuredReader for PDF files.
            reader = SimpleDirectoryReader(
                input_dir=data_path,
                file_extractor={".pdf": UnstructuredReader()}
            )
            documents = reader.load_data()
            
            print(f"--- Loaded {len(documents)} document(s) cleanly. Creating index... ---")
            instance_index = VectorStoreIndex.from_documents(documents)
            print("--- Indexing complete. ---")
            return instance_index

        task = asyncio.to_thread(load_and_index)
        index = await asyncio.wait_for(task, timeout=120.0) # Increased timeout slightly for the slower, smarter parser
        return {"status": "success", "message": f"File '{file.filename}' uploaded and indexed."}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Document processing timed out. The file may be extremely large.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/scrape_and_index")
async def scrape_and_index_url(request: ScrapeRequest):
    # This endpoint for URL scraping remains the same
    global index
    url = str(request.url)
    try:
        def scrape_and_load():
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text_content = soup.get_text(separator='\n', strip=True)
            documents = [Document(text=text_content, metadata={"source_url": url})]
            instance_index = VectorStoreIndex.from_documents(documents)
            return instance_index
        task = asyncio.to_thread(scrape_and_load)
        index = await asyncio.wait_for(task, timeout=90.0)
        return {"status": "success", "message": f"URL '{url}' scraped and indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/stream_query")
async def stream_query_document(request: QueryRequest):
    # This endpoint remains the same
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="No document indexed. Please upload or scrape first.")
    
    async def event_stream():
        def run_streaming_query():
            chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
            return chat_engine.stream_chat(request.question)
        try:
            streaming_response = await asyncio.to_thread(run_streaming_query)
            full_answer = ""
            for token in streaming_response.response_gen:
                delta = StreamResponse(answer_delta=token)
                yield f"data: {delta.json()}\n\n"
                full_answer += token
                await asyncio.sleep(0.01)
            
            source_nodes = []
            for node in streaming_response.source_nodes:
                file_name = node.metadata.get('file_name') or node.metadata.get('source_url', 'N/A')
                clean_file_name = os.path.basename(file_name)
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