import streamlit as st
import json
import networkx as nx
import os
import gc

# --- DISABLE TELEMETRY ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- IMPORTS ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
DATA_PATH = "/mnt/c/rag_data"
# Keep using the Graph DB path
DB_PATH = "/mnt/d/rag_data/chroma_graph_db"
MODEL_NAME = "phi3:mini" 

st.set_page_config(page_title="Local HippoRAG", layout="wide")
st.title("🕸️ Local Graph RAG (Gemini/Hippo Compatible)")

# --- 1. LOAD DATA ---
if not os.path.exists(DATA_PATH):
    st.error(f"Missing folder: {DATA_PATH}")
    st.stop()

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
selected_file = st.sidebar.selectbox("Select Knowledge Graph (JSON):", ["Select file..."] + files)

# --- 2. PARSER ---
def extract_triples_smart(data):
    """
    Hybrid parser:
    1. Looks specifically for the 'docs' -> 'extracted_triples' pattern (Gemini/Hippo).
    2. Falls back to generic recursive search if that structure isn't found.
    """
    triples = []
    
    # CASE A: HIPPO/GEMINI FORMAT (Docs List)
    # Structure: {'docs': [{'extracted_triples': [['Sub', 'Rel', 'Obj'], ...]}, ...]}
    if isinstance(data, dict) and 'docs' in data and isinstance(data['docs'], list):
        st.toast(f"Detected Gemini/Hippo format with {len(data['docs'])} chunks.")
        for doc in data['docs']:
            # Check for 'extracted_triples'
            if 'extracted_triples' in doc:
                raw_list = doc['extracted_triples']
                for item in raw_list:
                    # Item could be a list ['A', 'B', 'C'] or dict {'head':...}
                    if isinstance(item, list) and len(item) >= 3:
                        triples.append((str(item[0]), str(item[1]), str(item[2])))
                    elif isinstance(item, dict):
                        # Handle Dict inside list
                        s = item.get('subject') or item.get('head') or item.get('entity1') or item.get('source')
                        r = item.get('relation') or item.get('type') or item.get('label')
                        o = item.get('object') or item.get('tail') or item.get('entity2') or item.get('target')
                        if s and r and o:
                            triples.append((str(s), str(r), str(o)))
        return triples

    # CASE B: GENERIC RECURSIVE (Fallback)
    # If it's just a raw list of triples or a different format
    stack = [data]
    while stack:
        current = stack.pop()
        
        if isinstance(current, dict):
            s = current.get('subject') or current.get('head') or current.get('arg1')
            r = current.get('relation') or current.get('type') or current.get('predicate')
            o = current.get('object') or current.get('tail') or current.get('arg2')
            
            if s and r and o:
                triples.append((str(s), str(r), str(o)))
            
            for v in current.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                    
        elif isinstance(current, list):
            # Check for simple list triples ['A','B','C']
            if len(current) >= 3 and all(isinstance(x, str) for x in current[:3]):
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
        
        raw = extract_triples_smart(data)
        G = nx.DiGraph()
        txt_triples = []
        
        for s, r, o in raw:
            s, r, o = s.strip(), r.strip(), o.strip()
            if s and r and o:
                G.add_edge(s, o, relation=r)
                txt_triples.append(f"{s} --[{r}]--> {o}")
        return G, txt_triples, None

    with st.spinner("Parsing Graph Data..."):
        G, text_triples, error_msg = build_graph(file_path)
        
        if error_msg:
            st.error(f"Error: {error_msg}")
        elif not G or len(G.nodes()) == 0:
            st.warning("⚠️ No triples found.")
            st.info("The file structure looks correct (Docs -> Extracted Triples), but the lists appear to be empty. This implies the AI didn't extract any facts from the source text.")
        else:
            st.sidebar.success(f"Graph Loaded! Nodes: {len(G.nodes())}")
            
            # --- 3. INDEXING ---
            collection_name = f"graph_{selected_file.replace('.','_')}"
            embed = OllamaEmbeddings(model=MODEL_NAME)
            
            vector_store = Chroma(
                embedding_function=embed,
                collection_name=collection_name,
                persist_directory=DB_PATH
            )
            
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
                    nodes_to_process = nodes[existing_count:]
                    BATCH_SIZE = 50 
                    docs_buffer = []
                    
                    for i, node in enumerate(nodes_to_process):
                        docs_buffer.append(Document(page_content=node, metadata={"node": node}))
                        if len(docs_buffer) >= BATCH_SIZE or i == len(nodes_to_process) - 1:
                            try:
                                vector_store.add_documents(docs_buffer)
                                docs_buffer = [] 
                                current_idx = existing_count + i + 1
                                percent = min(1.0, current_idx / total_nodes)
                                progress_bar.progress(percent)
                                status_text.text(f"Indexing... {current_idx} / {total_nodes}")
                                if i % 1000 == 0: gc.collect()
                            except Exception as e:
                                st.error(f"Error: {e}")
                                break
                    st.success("Indexing Complete!")
                    st.rerun()
            else:
                st.success("✅ Index Ready!")
                st.session_state.vector_index = vector_store

    # --- 4. CHAT ---
    if "vector_index" in st.session_state:
        st.divider()
        query = st.chat_input("Ask the Graph...")
        if "messages" not in st.session_state: st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if query:
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("assistant"):
                with st.spinner("Reasoning..."):
                    try:
                        retriever = st.session_state.vector_index.as_retriever(search_kwargs={"k": 8})
                        results = retriever.invoke(query)
                        starts = [d.metadata['node'] for d in results]
                        facts = []
                        for node in starts:
                            if node in G:
                                for n, a in G[node].items(): facts.append(f"{node} -> {a.get('relation')} -> {n}")
                                for u, v, d in G.in_edges(node, data=True): facts.append(f"{u} -> {d.get('relation')} -> {node}")
                        context = "\n".join(list(set(facts))[:50])
                        llm = ChatOllama(model=MODEL_NAME)
                        prompt = f"Facts:\n{context}\n\nQuestion: {query}"
                        res = llm.invoke(prompt)
                        st.markdown(res.content)
                        st.session_state.messages.append({"role": "assistant", "content": res.content})
                    except Exception as e: st.error(f"Error: {e}")
