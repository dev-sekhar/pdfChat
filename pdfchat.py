import os
import_failed = False
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # from langchain_community.embeddings import HuggingFaceEmbeddings # Old import
    from langchain_huggingface import HuggingFaceEmbeddings # New recommended import
    # from langchain_community.llms import CTransformers # No longer using CTransformers
    from langchain_community.llms import LlamaCpp # Using LlamaCpp
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    import_failed = True
    print("ImportError: One or more required LangChain modules are missing.")
    print("Please ensure you have installed all necessary libraries:")
    print("pip install langchain faiss-cpu pypdf tiktoken langchain-community sentence-transformers llama-cpp-python langchain-huggingface")


def create_vector_db_from_pdf(pdf_path: str, persist_directory: str = "pdf_vector_db_faiss_local"):
    """
    Creates a FAISS vector database from a PDF file using local embeddings.
    If a persisted database exists, it loads it. Otherwise, it processes the PDF.
    """
    print("Initializing local embeddings (Sentence Transformers via langchain-huggingface)...")
    # Using a popular, good-quality, and relatively small model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have a GPU and PyTorch with CUDA
    )
    print("Local embeddings initialized.")

    if os.path.exists(persist_directory):
        print(f"Loading existing vector database from {persist_directory}...")
        try:
            # allow_dangerous_deserialization is needed for FAISS with custom embeddings/objects
            vector_db = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
            print("Vector database loaded successfully.")
            return vector_db
        except Exception as e:
            print(f"Error loading existing database: {e}. Recreating...")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None

    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        print("Error: Could not load any documents from the PDF.")
        return None

    print(f"Loaded {len(documents)} pages from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        print("Error: Text splitting resulted in no chunks.")
        return None

    print(f"Split PDF into {len(texts)} text chunks.")

    print("Creating vector database with local embeddings (this may take a few moments)...")
    vector_db = FAISS.from_documents(texts, embeddings)
    print("Vector database created successfully.")

    try:
        vector_db.save_local(persist_directory)
        print(f"Vector database saved to {persist_directory}")
    except Exception as e:
        print(f"Error saving vector database: {e}")

    return vector_db

def get_qa_chain(vector_db, model_path: str): # No longer need model_type for LlamaCpp
    """
    Creates a Question-Answering chain using the vector database and LlamaCpp.
    """
    print(f"Loading local LLM using LlamaCpp from: {model_path}")
    try:
        llm = LlamaCpp(
            model_path=model_path,    # Path to your GGUF model
            n_ctx=2048,               # Context window size, can increase if model supports
            n_batch=512,              # Batch size for prompt processing
            temperature=0.1,          # Lower for more deterministic output
            max_tokens=512,           # Max tokens to generate in the answer
            n_gpu_layers=0,           # Number of layers to offload to GPU (0 for CPU)
                                      # Set to -1 to try to offload all possible layers to GPU
            # use_mmap=True,          # Enable if you have memory issues and model is on disk
            verbose=True              # Set to True for detailed llama.cpp output during loading & inference
        )
        print("LlamaCpp LLM loaded successfully.")
    except Exception as e:
        print(f"Error loading LlamaCpp model: {e}")
        print("Make sure you have 'llama-cpp-python' installed: pip install llama-cpp-python")
        print("Also check the model path and integrity. If using GPU, check CUDA/ROCm setup.")
        return None

    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Provide a concise answer.

    Context:
    {context}

    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def main():
    """Main function to run the PDF chatbot with local models."""
    if import_failed:
        return

    pdf_path = input("Enter the path to your PDF file: ")
    if not pdf_path.lower().endswith(".pdf"):
        print("Error: Please provide a valid .pdf file.")
        return

    # --- Configuration for Local LLM ---
    local_llm_path = input("Enter the FULL path to your downloaded GGUF model file: ")
    if not os.path.exists(local_llm_path):
        print(f"Error: LLM model file not found at {local_llm_path}")
        print("Please download a GGUF model (e.g., from TheBloke on Hugging Face) and provide the correct path.")
        return

    # Model type is not explicitly needed for LlamaCpp as it infers from GGUF
    # model_type_input = input("Enter the model type (e.g., llama, mistral, phi-2, gemma): ").lower()
    # ... (removed model_type logic)

    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    persist_dir = f"{pdf_filename}_faiss_local_index"

    print("\nInitializing PDF Chatbot with Local Models...")
    vector_db = create_vector_db_from_pdf(pdf_path, persist_directory=persist_dir)

    if not vector_db:
        print("Failed to initialize vector database. Exiting.")
        return

    qa_chain = get_qa_chain(vector_db, model_path=local_llm_path) # Pass only model_path

    if not qa_chain:
        print("Failed to initialize QA chain. Exiting.")
        return

    print("\nPDF Chatbot is ready! Type 'exit' or 'quit' to end.")
    print("----------------------------------------------------")

    while True:
        query = input("Ask a question about the PDF: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        if not query.strip():
            continue

        try:
            print("Thinking (using local LlamaCpp LLM)...")
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "No answer found.")
            print(f"\nAnswer: {answer}")

            # source_documents = result.get("source_documents")
            # if source_documents:
            #     print("\nSources:")
            #     for i, doc in enumerate(source_documents):
            #         print(f"--- Source {i+1} (Page {doc.metadata.get('page', 'N/A')}) ---")
            #         print(doc.page_content[:200] + "...")
            #     print("----------------------------------------------------")

        except Exception as e:
            print(f"An error occurred during QA: {e}")
            print("Please try again or check your setup.")
        print("\n----------------------------------------------------")

if __name__ == "__main__":
    main()