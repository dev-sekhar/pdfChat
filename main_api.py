import os
import shutil
import uuid  # For unique session/job IDs
import time
import logging  # For better logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain and PDF Processing Imports (from your pdfchat.py) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# --- End LangChain Imports ---

# --- Global Variables and Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Directories
PDF_UPLOADS_DIR = "./pdf_uploads"
VECTOR_STORES_DIR = "./vector_stores"
LLM_MODELS_DIR = "./llm_models/"  # Directory for your GGUF models

os.makedirs(PDF_UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)
os.makedirs(LLM_MODELS_DIR, exist_ok=True)

# In-memory "databases" for state management
# For production, these should be replaced with persistent solutions like Redis or a proper database.
# Tracks status of PDF processing jobs {pdf_id: {"status": "...", "message": "..."}}
PROCESSING_STATUS = {}
# Caches active QA chains { "pdf_id_llm_filename": qa_chain_object }
ACTIVE_QA_CHAINS = {}
LOADED_EMBEDDINGS = None  # Global variable to hold the initialized embedding model
LOADED_VECTOR_DBS = {}   # Caches loaded vector DBs { persist_dir_path: faiss_object }

# --- Core PDF Processing and LangChain Logic (Adapted from your pdfchat.py) ---


def initialize_embeddings_model():
    """Initializes and returns the HuggingFace embedding model."""
    global LOADED_EMBEDDINGS
    if LOADED_EMBEDDINGS is None:
        logger.info("Initializing HuggingFace embeddings model...")
        try:
            LOADED_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                # Or 'cuda' if GPU is available and configured
                model_kwargs={'device': 'cpu'}
            )
            logger.info(
                "HuggingFace embeddings model initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize embeddings model: {e}", exc_info=True)
            raise  # Re-raise the exception to signal failure
    return LOADED_EMBEDDINGS


def actual_create_vector_db_from_pdf(pdf_file_path: str, persist_dir: str):
    """
    Processes a PDF: extracts text, chunks, creates embeddings, and saves a FAISS vector store.
    This function contains the core logic from your `create_vector_db_from_pdf`.
    """
    logger.info(f"Starting to process PDF: {pdf_file_path} into {persist_dir}")
    embeddings = initialize_embeddings_model()  # Ensure embeddings are loaded

    # Check if already processed (though background task usually handles new ones)
    if os.path.exists(os.path.join(persist_dir, "done.flag")):  # Using a flag file
        logger.info(
            f"Vector store already exists at {persist_dir}, skipping creation.")
        # Optionally load and return it, but background task usually creates new
        return

    # Load PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"Could not load any documents from PDF: {pdf_file_path}")
    logger.info(f"Loaded {len(documents)} pages from {pdf_file_path}.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError(
            f"Text splitting resulted in no chunks for PDF: {pdf_file_path}")
    logger.info(f"Split PDF into {len(texts)} text chunks.")

    # Create FAISS vector store
    logger.info(f"Creating FAISS vector store for {pdf_file_path}...")
    vector_db = FAISS.from_documents(texts, embeddings)
    logger.info(f"FAISS vector store created.")

    # Save vector store
    os.makedirs(persist_dir, exist_ok=True)  # Ensure directory exists
    vector_db.save_local(persist_dir)
    with open(os.path.join(persist_dir, "done.flag"), "w") as f:  # Create a flag file
        f.write("processed")
    logger.info(f"Vector store saved to {persist_dir}")
    # No need to return vector_db here, as it will be loaded on demand by get_qa_chain_for_pdf


def get_vector_db(persist_dir: str):
    """Loads a FAISS vector database from the specified directory."""
    global LOADED_VECTOR_DBS
    if persist_dir in LOADED_VECTOR_DBS:
        return LOADED_VECTOR_DBS[persist_dir]

    if not os.path.exists(os.path.join(persist_dir, "done.flag")):
        raise FileNotFoundError(
            f"Vector store not found or not fully processed at {persist_dir}")

    logger.info(f"Loading vector DB from {persist_dir}...")
    embeddings = initialize_embeddings_model()
    try:
        vector_db = FAISS.load_local(
            persist_dir, embeddings, allow_dangerous_deserialization=True)
        LOADED_VECTOR_DBS[persist_dir] = vector_db  # Cache it
        logger.info(f"Vector DB loaded successfully from {persist_dir}")
        return vector_db
    except Exception as e:
        logger.error(
            f"Failed to load vector DB from {persist_dir}: {e}", exc_info=True)
        raise


def get_qa_chain_for_pdf(persist_dir_for_pdf: str, llm_model_filename: str):
    """
    Creates or retrieves a cached QA chain for a specific PDF and LLM model.
    """
    cache_key = f"{persist_dir_for_pdf}_{llm_model_filename}"
    if cache_key in ACTIVE_QA_CHAINS:
        logger.info(f"Returning cached QA chain for {cache_key}")
        return ACTIVE_QA_CHAINS[cache_key]

    logger.info(
        f"QA chain for {cache_key} not found in cache. Creating new one.")

    # 1. Load Vector DB
    vector_db = get_vector_db(persist_dir_for_pdf)

    # 2. Load LLM
    model_path = os.path.join(LLM_MODELS_DIR, llm_model_filename)
    if not os.path.exists(model_path):
        logger.error(f"LLM model file not found: {model_path}")
        raise FileNotFoundError(
            f"LLM model '{llm_model_filename}' not found in server's model directory.")

    logger.info(f"Loading LLM: {model_path}")
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            n_batch=512,
            temperature=0.1,
            max_tokens=512,
            n_gpu_layers=0,  # CPU by default
            verbose=False  # Keep False for API, True for debugging in llama.cpp
        )
        logger.info(f"LLM {llm_model_filename} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLM {model_path}: {e}", exc_info=True)
        raise  # Propagate error

    # 3. Define Prompt Template
    prompt_template_str = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Provide a concise answer.

Context:
{context}

Question: {question}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template_str
    )

    # 4. Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,  # Usually False for API to reduce payload size
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    ACTIVE_QA_CHAINS[cache_key] = qa_chain  # Cache the created chain
    logger.info(f"QA chain for {cache_key} created and cached.")
    return qa_chain


# --- FastAPI App Setup ---
app = FastAPI(title="PDF Chatbot API")

# Configure CORS for frontend access (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Request/Response Schemas ---


class ProcessResponse(BaseModel):
    pdf_id: str
    message: str
    status_url: str  # URL to poll for processing status


class StatusResponse(BaseModel):
    pdf_id: str
    status: str  # e.g., "queued", "processing", "completed", "failed"
    message: str | None = None


class QueryRequest(BaseModel):
    pdf_id: str
    query: str
    llm_model_filename: str  # Filename of the GGUF model, e.g., "phi-2.Q4_K_M.gguf"


class QueryResponse(BaseModel):
    pdf_id: str
    answer: str
    # sources: list | None = None # Optional: if you want to return source document snippets


class AvailableLLM(BaseModel):
    filename: str

# --- Background PDF Processing Task ---


def background_pdf_processing_task(pdf_id: str, temp_pdf_path: str, persist_dir_for_pdf: str):
    """
    The actual function run by BackgroundTasks to process the PDF.
    Updates PROCESSING_STATUS.
    """
    try:
        logger.info(
            f"[{pdf_id}] Background task started for PDF: {temp_pdf_path}")
        PROCESSING_STATUS[pdf_id] = {
            "status": "processing", "message": "Extracting text, creating embeddings..."}

        actual_create_vector_db_from_pdf(
            temp_pdf_path, persist_dir_for_pdf)  # Call the core logic

        PROCESSING_STATUS[pdf_id] = {
            "status": "completed", "message": "PDF processed successfully. Ready for querying."}
        logger.info(
            f"[{pdf_id}] Background task completed for PDF: {temp_pdf_path}")
    except Exception as e:
        logger.error(
            f"[{pdf_id}] Error during background PDF processing: {e}", exc_info=True)
        PROCESSING_STATUS[pdf_id] = {
            "status": "failed", "message": f"Processing error: {str(e)}"}
    finally:
        # Clean up the temporarily uploaded PDF file
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info(
                    f"[{pdf_id}] Cleaned up temporary PDF file: {temp_pdf_path}")
            except Exception as e_rem:
                logger.error(
                    f"[{pdf_id}] Error cleaning up temp PDF {temp_pdf_path}: {e_rem}")

# --- API Endpoints ---


@app.on_event("startup")
async def on_api_startup():
    """
    Actions to perform when the API server starts.
    Pre-initializes the embedding model.
    """
    logger.info("API server starting up...")
    try:
        initialize_embeddings_model()
    except Exception as e:
        logger.critical(
            f"CRITICAL: Failed to initialize embeddings model on startup. API might not function correctly. Error: {e}", exc_info=True)
        # Depending on severity, you might want the app to not start or enter a degraded mode.
    logger.info("API startup complete.")


@app.post("/upload_pdf/", response_model=ProcessResponse)
async def api_upload_and_process_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF. The PDF is saved temporarily and processed in the background.
    Returns a pdf_id and a URL to check the processing status.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    # Generate a unique ID for this PDF processing job
    pdf_id = str(uuid.uuid4())
    original_filename_base = os.path.splitext(file.filename)[0]

    # Sanitize filename for use in directory path (simple version)
    safe_filename_base = "".join(c if c.isalnum() or c in (
        ' ', '_', '-') else '_' for c in original_filename_base).rstrip()

    # Path where the uploaded PDF will be temporarily saved
    temp_pdf_path = os.path.join(PDF_UPLOADS_DIR, f"{pdf_id}_{file.filename}")
    # Directory where the FAISS index for this PDF will be persisted
    persist_dir_for_pdf = os.path.join(
        VECTOR_STORES_DIR, f"{pdf_id}_{safe_filename_base}_faiss_index")

    # Save the uploaded file to the temporary path
    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(
            f"[{pdf_id}] PDF '{file.filename}' uploaded and saved to {temp_pdf_path}")
    except Exception as e:
        logger.error(
            f"[{pdf_id}] Failed to save uploaded PDF '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded PDF: {str(e)}")
    finally:
        await file.close()  # Ensure the file stream is closed

    # Add the PDF processing to background tasks
    background_tasks.add_task(
        background_pdf_processing_task, pdf_id, temp_pdf_path, persist_dir_for_pdf)
    PROCESSING_STATUS[pdf_id] = {"status": "queued",
                                 "message": "PDF processing has been queued."}

    return ProcessResponse(
        pdf_id=pdf_id,
        message="PDF upload successful. Processing has started in the background.",
        status_url=f"/process_status/{pdf_id}"  # Client can poll this URL
    )


@app.get("/process_status/{pdf_id}", response_model=StatusResponse)
async def api_get_process_status(pdf_id: str):
    """
    Endpoint to check the processing status of a previously uploaded PDF.
    """
    status_info = PROCESSING_STATUS.get(pdf_id)
    if not status_info:
        # If not in memory, check if it was processed on disk (e.g., after server restart)
        # This requires finding the directory by pdf_id prefix.
        found_dirs = [d for d in os.listdir(VECTOR_STORES_DIR) if d.startswith(
            f"{pdf_id}_") and os.path.isdir(os.path.join(VECTOR_STORES_DIR, d))]
        if found_dirs and os.path.exists(os.path.join(VECTOR_STORES_DIR, found_dirs[0], "done.flag")):
            return StatusResponse(pdf_id=pdf_id, status="completed", message="PDF was processed previously (found on disk).")
        raise HTTPException(
            status_code=404, detail=f"Processing status for PDF ID '{pdf_id}' not found.")
    return StatusResponse(pdf_id=pdf_id, status=status_info["status"], message=status_info.get("message"))


@app.get("/available_llms/", response_model=list[AvailableLLM])
async def api_get_available_llms():
    """
    Endpoint to list available GGUF LLM models from the LLM_MODELS_DIR.
    """
    try:
        if not os.path.exists(LLM_MODELS_DIR) or not os.path.isdir(LLM_MODELS_DIR):
            logger.warning(
                f"LLM models directory '{LLM_MODELS_DIR}' not found.")
            return []
        gguf_files = [f for f in os.listdir(
            LLM_MODELS_DIR) if f.lower().endswith(".gguf")]
        if not gguf_files:
            logger.info(f"No GGUF models found in {LLM_MODELS_DIR}")
        return [AvailableLLM(filename=f) for f in gguf_files]
    except Exception as e:
        logger.error(
            f"Error listing LLM models from '{LLM_MODELS_DIR}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Could not retrieve list of LLM models.")


@app.post("/query/", response_model=QueryResponse)
async def api_query_pdf(request: QueryRequest):
    """
    Endpoint to ask a question about a processed PDF.
    Requires pdf_id, query, and the filename of the LLM to use.
    """
    # 1. Check PDF processing status
    status_info = PROCESSING_STATUS.get(request.pdf_id)
    is_processed_on_disk = False
    actual_persist_dir = None

    found_dirs = [d for d in os.listdir(VECTOR_STORES_DIR) if d.startswith(
        f"{request.pdf_id}_") and os.path.isdir(os.path.join(VECTOR_STORES_DIR, d))]
    if found_dirs and os.path.exists(os.path.join(VECTOR_STORES_DIR, found_dirs[0], "done.flag")):
        is_processed_on_disk = True
        actual_persist_dir = os.path.join(VECTOR_STORES_DIR, found_dirs[0])

    if not ((status_info and status_info.get("status") == "completed") or is_processed_on_disk):
        current_status_msg = status_info.get('status') if status_info else (
            'not found on disk' if not is_processed_on_disk else 'unknown')
        raise HTTPException(
            status_code=400, detail=f"PDF '{request.pdf_id}' is not yet processed or processing failed. Status: {current_status_msg}")

    if not actual_persist_dir:  # Should have been set if is_processed_on_disk or status is completed
        raise HTTPException(
            status_code=500, detail="Internal error: Could not determine persistence directory for processed PDF.")

    # 2. Get or create the QA chain
    try:
        qa_chain = get_qa_chain_for_pdf(
            actual_persist_dir, request.llm_model_filename)
    except FileNotFoundError as e:  # LLM model file not found
        logger.error(f"[{request.pdf_id}] Query failed: {e}",
                     exc_info=True)  # exc_info only if debugging
        # Pass specific error to client
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"[{request.pdf_id}] Query failed due to QA chain initialization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not initialize QA system: {str(e)}")

    # 3. Perform the query
    try:
        logger.info(
            f"[{request.pdf_id}] Processing query: '{request.query[:50]}...' with LLM '{request.llm_model_filename}'")
        start_time = time.time()

        # This is the actual call to your LangChain QA chain
        result = qa_chain.invoke({"query": request.query})
        answer = result.get(
            "result", "No answer found in the provided context.")

        end_time = time.time()
        logger.info(
            f"[{request.pdf_id}] Query processed in {end_time - start_time:.2f}s. Answer: '{answer[:50]}...'")

        return QueryResponse(pdf_id=request.pdf_id, answer=answer)
    except Exception as e:
        logger.error(
            f"[{request.pdf_id}] Error during query processing with LLM: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing query with LLM: {str(e)}")

# --- To run the FastAPI backend ---
# 1. Save this file as main_api.py
# 2. Ensure you have a directory named 'llm_models' in the same location, and place your .gguf files there.
# 3. In your terminal, navigate to the directory containing main_api.py.
# 4. Run the Uvicorn server:
#    uvicorn main_api:app --reload
#
# The API will be accessible at http://127.0.0.1:8000
# API documentation (Swagger UI) will be at http://127.0.0.1:8000/docs
# API documentation (ReDoc) will be at http://127.0.0.1:8000/redoc
# end of file
