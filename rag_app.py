import streamlit as st
import os
import shutil

# --- DISABLE TELEMETRY ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# We use the "Classic" RetrievalQA because it is crash-proof with local models
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
DATA_PATH = "/mnt/c/rag_data"
MODEL_NAME = "phi3:mini"

st.set_page_config(page_title="Local RAG Station", layout="wide")
st.title(f"⚡ Local RAG (Model: {MODEL_NAME})")

# --- 1. SIDEBAR ---
st.sidebar.header("1. Select Data Sources")

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find folder: {DATA_PATH}")
    st.stop()

files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

selected_files = st.sidebar.multiselect(
    "Pick files (PDF, DOCX, CSV):", 
    files,
    default=[],
    help="Select one or more files to ingest."
)

# --- 2. PROCESSING ---
if selected_files:
    # Check if we need to rebuild the DB
    current_selection_set = set(selected_files)
    previous_selection_set = set(st.session_state.get("current_files", []))

    if current_selection_set != previous_selection_set:
        
        status_text = st.sidebar.empty()
        all_documents = []
        
        # Load Files
        for filename in selected_files:
            file_path = os.path.join(DATA_PATH, filename)
            status_text.text(f"Loading {filename}...")
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    all_documents.extend(loader.load())
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    all_documents.extend(loader.load())
                elif filename.endswith(".csv"):
                    loader = CSVLoader(file_path)
                    all_documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")

        if all_documents:
            with st.spinner(f"Reading with {MODEL_NAME}..."):
                # A. Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(all_documents)

                # B. Embed & Store (In-Memory)
                embeddings = OllamaEmbeddings(model=MODEL_NAME)
                
                vector_db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    collection_name="local_rag_memory"
                )
                
                st.session_state.retriever = vector_db.as_retriever()
                st.session_state.current_files = selected_files
                st.session_state.messages = [] 
                status_text.success("Knowledge Base Ready!")
        else:
            st.warning("No text found in selected files.")

# --- 3. CHAT ---
if "retriever" in st.session_state and selected_files:
    st.divider()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatOllama(model=MODEL_NAME)
                    
                    # Classic RetrievalQA Chain
                    # This handles the prompt and LLM connection internally
                    # preventing the "NoneType" error you saw.
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.retriever,
                        return_source_documents=False
                    )
                    
                    # Run the query
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info("👈 Select files to begin.")
