import streamlit as st
import json
import networkx as nx
import os
import gc # Garbage Collection to manage RAM

# --- DISABLE TELEMETRY ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- IMPORTS ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
# Source JSONs are on C:
DATA_PATH = "/mnt/c/rag_data"

# SAVE THE DATABASE TO D (Graph Specific Folder):
DB_PATH = "/mnt/d/rag_data/chroma_graph_db"

MODEL_NAME = "phi3:mini" 

st.set_page_config(page_title="Local HippoRAG", layout="wide")
st.title("🕸️ Local Graph RAG (Resumable & Persistent)")

# --- 1. LOAD DATA ---
if not os.path.exists(DATA_PATH):
    st.error(f"Missing folder: {DATA_PATH}")
    st.stop()

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
selected_file = st.sidebar.selectbox("Select Knowledge Graph (JSON):", ["Select file..."] + files)

# --- 2. ITERATIVE PARSER (Crash-Proof) ---
def extract_triples_iterative(data):
    """
    Finds triples using a stack loop instead of recursion.
    This prevents 'RecursionError' on deep files.
    """
    triples = []
    stack = [data]
    
    while stack:
        current = stack.pop()
        
        if isinstance(current, dict):
            # Check for Triple Structure
            sub = current.get('subject') or current.get('head') or current.get('arg1')
            rel = current.get('relation') or current.get('type') or current.get('predicate')
            obj = current.get('object') or current.get('tail') or current.get('arg2')
            
            if sub and rel and obj:
                triples.append((str(sub), str(rel), str(obj)))
            
            # Continue searching deeper
            for v in current.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                    
        elif isinstance(current, list):
            # Check for list triples ['A','B','C']
            if len(current) == 3 and all(isinstance(x, str) for x in current):
                triples.append((current[0], current[1], current[2]))
            else:
                for item in current:
                    if isinstance(item, (dict, list)):
                        stack.append(item)
    return triples

if selected_file and selected_file != "Select file...":
    file_path = os.path.join(DATA_PATH, selected_file)
    
    @st.cache_resource
    def build_graph(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return None, [], str(e)
        
        raw = extract_triples_iterative(data)
        G = nx.DiGraph()
        txt_triples = []
        
        for s, r, o in raw:
            s, r, o = s.strip(), r.strip(), o.strip()
            if s and r and o:
                G.add_edge(s, o, relation=r)
                txt_triples.append(f"{s} --[{r}]--> {o}")
        return G, txt_triples, None

    with st.spinner("Loading Graph Structure..."):
        G, text_triples, error_msg = build_graph(file_path)
        
        if error_msg:
            st.error(f"Error: {error_msg}")
        elif not G or len(G.nodes()) == 0:
            st.error("No triples found in this file.")
        else:
            st.sidebar.success(f"Graph Structure Loaded! Nodes: {len(G.nodes())}")
            
            # --- 3. RESUMABLE INDEXING ---
            # Define the Persistent DB on D: Drive
            collection_name = f"graph_{selected_file.replace('.','_')}"
            embed = OllamaEmbeddings(model=MODEL_NAME)
            
            # Initialize connection to Disk
            vector_store = Chroma(
                embedding_function=embed,
                collection_name=collection_name,
                persist_directory=DB_PATH
            )
            
            # Check existing progress
            try:
                existing_count = vector_store._collection.count()
            except:
                existing_count = 0
            
            nodes = list(G.nodes())
            total_nodes = len(nodes)
            
            st.write(f"**Index Status:** {existing_count} / {total_nodes} nodes indexed.")
            
            if existing_count < total_nodes:
                if st.button(f"Resume Indexing ({total_nodes - existing_count} remaining)"):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Skip already processed nodes
                    nodes_to_process = nodes[existing_count:]
                    
                    # Batch Settings
                    BATCH_SIZE = 50 
                    docs_buffer = []
                    
                    for i, node in enumerate(nodes_to_process):
                        docs_buffer.append(Document(page_content=node, metadata={"node": node}))
                        
                        # Process Batch
                        if len(docs_buffer) >= BATCH_SIZE or i == len(nodes_to_process) - 1:
                            try:
                                vector_store.add_documents(docs_buffer)
                                docs_buffer = [] # Clear buffer
                                
                                # Visual Update
                                current_idx = existing_count + i + 1
                                percent = min(1.0, current_idx / total_nodes)
                                progress_bar.progress(percent)
                                status_text.text(f"Indexing... {current_idx} / {total_nodes}")
                                
                                # Free RAM
                                if i % 1000 == 0:
                                    gc.collect()
                                    
                            except Exception as e:
                                st.error(f"Crash at node {node}: {e}")
                                break
                    
                    st.success("Indexing Complete!")
                    st.rerun()
            else:
                st.success("✅ Index is fully built on D: drive!")
                st.session_state.vector_index = vector_store

    # --- 4. CHAT INTERFACE ---
    if "vector_index" in st.session_state:
        st.divider()
        query = st.chat_input("Ask a question about the graph relationships...")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if query:
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("assistant"):
                with st.spinner("Connecting the dots..."):
                    try:
                        # 1. Search for Entry Nodes
                        retriever = st.session_state.vector_index.as_retriever(search_kwargs={"k": 10})
                        results = retriever.invoke(query)
                        starts = [d.metadata['node'] for d in results]
                        
                        # 2. Traverse Graph (Neighbors)
                        facts = []
                        for node in starts:
                            if node in G:
                                # Outgoing edges
                                for nbr, attr in G[node].items():
                                    facts.append(f"{node} -> {attr.get('relation')} -> {nbr}")
                                # Incoming edges
                                for u, v, d in G.in_edges(node, data=True):
                                    facts.append(f"{u} -> {d.get('relation')} -> {node}")
                        
                        # Dedup and limit context
                        uniq_facts = list(set(facts))[:60]
                        context = "\n".join(uniq_facts)
                        
                        # 3. Generate Answer
                        llm = ChatOllama(model=MODEL_NAME)
                        prompt = f"""Use these Graph Facts to answer the question.
                        
                        FACTS:
                        {context}
                        
                        QUESTION: {query}
                        """
                        res = llm.invoke(prompt)
                        
                        st.markdown(res.content)
                        st.session_state.messages.append({"role": "assistant", "content": res.content})
                        
                        with st.expander("See Graph Path"):
                            st.write(uniq_facts)
                    except Exception as e:
                        st.error(f"Error: {e}")
