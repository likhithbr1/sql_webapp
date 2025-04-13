import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sqlalchemy import create_engine, text

# ------------------------------
# Configuration – update these paths and model names
# ------------------------------
INDEX_PATH = "schema_index/faiss_index.bin"
META_PATH = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
LLAMA_MODEL_PATH = "sqlcoder-7b-2.Q4_K_M.gguf"   # your GGUF file for SQLCoder
TOP_K = 3                      # top-K tables from semantic search
N_CTX = 2048                   # SQLCoder context window
N_THREADS = 6                  # adjust for your CPU
DB_URI = "mysql+pymysql://root:admin@localhost/chatbot"

# ------------------------------
# Load FAISS index and table metadata
# ------------------------------
def load_faiss_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta

def build_reverse_fk_map(metadata: dict) -> dict:
    rev_map = {m["table_name"]: set() for m in metadata.values()}
    fk_pattern = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)
    for m in metadata.values():
        table = m["table_name"]
        ddl = m["create_stmt"]
        for ref in fk_pattern.findall(ddl):
            if ref in rev_map:
                rev_map[ref].add(table)
    return rev_map

def parse_forward_fks(ddl: str) -> set:
    fk_pattern = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)
    return set(fk_pattern.findall(ddl))

def semantic_search(query: str, embed_model, faiss_index, top_k: int):
    q_emb = embed_model.encode(query)
    q_emb = np.array([q_emb], dtype="float32")
    _, I = faiss_index.search(q_emb, top_k)
    return I[0]

def expand_with_related(idx_list, metadata, rev_fk_map):
    tables = {metadata[str(i)]["table_name"] for i in idx_list}
    extra = set()
    for i in idx_list:
        m = metadata[str(i)]
        table = m["table_name"]
        ddl = m["create_stmt"]
        extra.update(parse_forward_fks(ddl))
        extra.update(rev_fk_map.get(table, set()))
    return tables.union(extra)

def build_schema_snippet(table_names: set, metadata: dict) -> str:
    ddl_list = []
    # Preserve original order (if desired)
    for meta in metadata.values():
        if meta["table_name"] in table_names:
            ddl_list.append(meta["create_stmt"])
    return "\n\n".join(ddl_list)

def get_relevant_schema_snippet(query: str, embed_model, faiss_index, metadata, rev_fk_map, top_k: int) -> str:
    idxs = semantic_search(query, embed_model, faiss_index, top_k)
    final_tables = expand_with_related(idxs, metadata, rev_fk_map)
    schema_text = build_schema_snippet(final_tables, metadata)
    return schema_text

# ------------------------------
# Custom LLM Wrapper for SQLCoder (using your Llama model)
# ------------------------------
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Mapping, Any
import requests

class SQLCoderLLM(LLM):
    """Custom LLM wrapper for the SQLCoder model hosted locally/in your environment."""
    model_endpoint: str = ""  # Not used here because we call our local Llama model directly.
    api_token: str = ""       # Not used here; we use a local model.
    
    def _llm_type(self) -> str:
        return "sqlcoder"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # In our case, _llm is the Llama SQLCoder model instance loaded below.
        # We assume that _llm.create_completion yields chunks of generated text.
        final_text = ""
        for chunk in _llm.create_completion(
            prompt,
            max_tokens=512,
            stop=stop or ["```"],
            temperature=0.1,
            stream=False  # Change to True if you want streaming support.
        ):
            final_text += chunk["choices"][0]["text"]
        return final_text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": LLAMA_MODEL_PATH}

# ------------------------------
# Load Llama-based SQLCoder Model
# ------------------------------
def load_llama(model_path: str):
    print(f"Loading SQLCoder model from {model_path} …")
    return Llama(
        model_path=model_path,
        n_gpu_layers=0,       # CPU only; adjust if GPU is available
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=True,
        n_batch=512,
        use_mlock=True,
        use_mmap=True,
        logits_all=False,
    )

# ------------------------------
# Agent Prompt Template
# ------------------------------
CUSTOM_PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer the following question:
{question}

### Relevant Database Schema
{schema}

### SQL Query
```sql
"""

# ------------------------------
# Initialization of Global Components
# ------------------------------
def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model, _llm, _engine, sqlcoder_llm
    print("Initializing FAISS index, embeddings, Llama (SQLCoder), and database connection …")
    _faiss_index, _metadata = load_faiss_and_metadata(INDEX_PATH, META_PATH)
    _rev_fk_map = build_reverse_fk_map(_metadata)
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _llm = load_llama(LLAMA_MODEL_PATH)  # Load your SQLCoder Llama model
    _engine = create_engine(DB_URI)
    
    # Initialize our custom LLM wrapper for SQLCoder
    sqlcoder_llm = SQLCoderLLM()
    print("Models and DB connection loaded!")
    return sqlcoder_llm

# ------------------------------
# --- Integration with LangChain SQL Agent ---
# We'll use LangChain's create_sql_agent and the RunnablePassthrough
# to inject our custom schema retrieval.
# ------------------------------

# Import necessary LangChain modules:
from operator import itemgetter
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough

# Custom schema chain: given an input dict with "question", return a schema snippet.
def custom_schema_chain(input_dict: dict) -> str:
    question = input_dict["question"]
    schema_snippet = get_relevant_schema_snippet(question, _embed_model, _faiss_index, _metadata, _rev_fk_map, TOP_K)
    return schema_snippet

# ------------------------------
# Build the final agent chain:
# ------------------------------
def build_agent_chain(sqlcoder_llm) -> any:
    # Create the SQL query chain using our custom prompt template:
    query_chain = create_sql_agent(
        llm=sqlcoder_llm,
        db=SQLDatabase.from_uri(DB_URI),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    # Wrap our custom schema chain so that it takes the "question" input
    table_chain = {"input": itemgetter("question")} | custom_schema_chain

    # Use RunnablePassthrough to assign the dynamically obtained schema to the parameter "table_names_to_use"
    # (In our custom prompt, we expect a placeholder {schema}; the agent will substitute this value.)
    full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    return full_chain

# ------------------------------
# Main function to process a question via the agent
# ------------------------------
def process_question_agentic(question: str) -> dict:
    try:
        agent_chain = build_agent_chain(sqlcoder_llm)
        result = agent_chain.invoke({"question": question})
        return result
    except Exception as e:
        return {
            "sql": None,
            "results": [],
            "error": str(e)
        }

# ------------------------------
# Main script usage
# ------------------------------
if __name__ == "__main__":
    # Initialize models, schema tools, DB connection, and our custom SQLCoder LLM wrapper
    sqlcoder_llm = init_models()
    
    # Example user question
    user_question = "What are the top 5 orders by revenue in 2022?"
    
    # Process the question using the agentic chain
    answer = process_question_agentic(user_question)
    
    print("Generated SQL:\n", answer.get("sql"))
    print("Query Results:\n", answer.get("results"))
    if "error" in answer:
        print("Error:", answer.get("error"))
