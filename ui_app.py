import streamlit as st
import requests  # To make API calls
import time
import os

# --- Configuration for the API Backend ---
API_BASE_URL = "http://127.0.0.1:8000"  # Your FastAPI backend URL

# --- Helper Functions to Call API ---


def get_available_llms():
    try:
        response = requests.get(f"{API_BASE_URL}/available_llms/")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching LLM list: {e}")
        return []


def upload_pdf_to_api(uploaded_file_obj):
    files = {'file': (uploaded_file_obj.name,
                      uploaded_file_obj.getvalue(), uploaded_file_obj.type)}
    try:
        response = requests.post(f"{API_BASE_URL}/upload_pdf/", files=files)
        response.raise_for_status()  # Raise an exception for bad status codes
        # {"pdf_id": "...", "message": "...", "status_url": "..."}
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading PDF: {e}")
        return None


def get_pdf_status_from_api(pdf_id):
    try:
        response = requests.get(f"{API_BASE_URL}/process_status/{pdf_id}")
        response.raise_for_status()
        # {"pdf_id": "...", "status": "...", "message": "..."}
        return response.json()
    except requests.exceptions.RequestException as e:
        # It's okay if a 404 happens initially if status isn't there yet
        if response.status_code == 404:
            return {"pdf_id": pdf_id, "status": "unknown", "message": "Status not yet available."}
        st.error(f"Error getting PDF status: {e}")
        return None


def query_api(pdf_id, query_text, llm_model_filename):
    payload = {
        "pdf_id": pdf_id,
        "query": query_text,
        "llm_model_filename": llm_model_filename
    }
    try:
        response = requests.post(f"{API_BASE_URL}/query/", json=payload)
        response.raise_for_status()
        return response.json()  # {"pdf_id": "...", "answer": "..."}
    except requests.exceptions.RequestException as e:
        st.error(
            f"Error querying API: {e}. Response: {response.text if response else 'No response'}")
        return None


# --- Streamlit App ---
st.set_page_config(page_title="PDF Chat UI", layout="wide")
st.title("📄 Chat with your PDF (via API)")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Please upload a PDF and select an LLM to get started."}]
if "current_pdf_id" not in st.session_state:
    st.session_state.current_pdf_id = None
if "pdf_processing_complete" not in st.session_state:
    st.session_state.pdf_processing_complete = False
if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = None
if "available_llms" not in st.session_state:
    st.session_state.available_llms = get_available_llms()


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "1. Upload your PDF", type="pdf", key="pdf_uploader")

    if st.session_state.available_llms:
        llm_filenames = [llm['filename']
                         for llm in st.session_state.available_llms]
        st.session_state.selected_llm = st.selectbox(
            "2. Select LLM Model",
            options=llm_filenames,
            index=llm_filenames.index(
                st.session_state.selected_llm) if st.session_state.selected_llm and st.session_state.selected_llm in llm_filenames else 0,
            key="llm_selector"
        )
    else:
        st.warning("No LLM models found or API unreachable. Check backend.")
        st.session_state.selected_llm = None  # Ensure it's None if no models

    if st.button("Process PDF", key="process_button"):
        if uploaded_file is not None and st.session_state.selected_llm:
            st.session_state.current_pdf_id = None  # Reset
            st.session_state.pdf_processing_complete = False
            st.session_state.messages = [
                {"role": "assistant", "content": f"Uploading {uploaded_file.name}..."}]

            upload_response = upload_pdf_to_api(uploaded_file)
            if upload_response and "pdf_id" in upload_response:
                st.session_state.current_pdf_id = upload_response["pdf_id"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"PDF '{uploaded_file.name}' uploaded (ID: {st.session_state.current_pdf_id}). Processing in background..."})
                st.info(
                    f"PDF ID: {st.session_state.current_pdf_id}. Waiting for processing to complete...")
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "PDF upload failed."})
        else:
            st.warning("Please upload a PDF and select an LLM model.")

# --- Polling for PDF Processing Status ---
if st.session_state.current_pdf_id and not st.session_state.pdf_processing_complete:
    with st.spinner("Waiting for PDF processing to complete..."):
        while True:
            status_response = get_pdf_status_from_api(
                st.session_state.current_pdf_id)
            if status_response:
                current_status = status_response.get("status")
                status_message = status_response.get("message", "")
                # st.sidebar.write(f"Status: {current_status} - {status_message}") # For debugging
                if current_status == "completed":
                    st.session_state.pdf_processing_complete = True
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "PDF processing complete! You can now ask questions."})
                    st.sidebar.success("PDF Processed and Ready!")
                    st.experimental_rerun()  # Rerun to update main UI
                    break
                elif current_status == "failed":
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"PDF processing failed: {status_message}"})
                    st.sidebar.error(f"Processing failed: {status_message}")
                    st.session_state.current_pdf_id = None  # Reset
                    break
                elif current_status in ["queued", "processing"]:
                    # Keep polling
                    pass
                else:  # Unknown or error state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"PDF processing status unknown or error: {status_message}"})
                    st.sidebar.warning(f"Unknown status: {current_status}")
                    st.session_state.current_pdf_id = None  # Reset
                    break
            else:  # API error while getting status
                st.session_state.current_pdf_id = None  # Reset
                break  # Stop polling on API error
            time.sleep(3)  # Poll every 3 seconds


# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.current_pdf_id or not st.session_state.pdf_processing_complete:
        st.error("Please upload and wait for a PDF to be processed first.")
    elif not st.session_state.selected_llm:
        st.error("Please select an LLM model from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Assistant is thinking..."):
                api_response = query_api(
                    st.session_state.current_pdf_id, prompt, st.session_state.selected_llm)

            if api_response and "answer" in api_response:
                full_response = api_response["answer"]
            else:
                full_response = "Sorry, there was an error getting an answer from the API."

            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
