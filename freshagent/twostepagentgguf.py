# agent_vector_sql.py
"""
Run a LangChain SQL agent that first narrows the schema with FAISS
and then lets SQLCoder generate / execute the query.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import json, re, numpy as np, faiss
from typing import Optional, List, Any, Set
from sentence_transformers import SentenceTransformer
from sqlalchemy import inspect, create_engine, text
from llama_cpp import Llama

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

# ---------------------------------------------------------------------
# Paths / hyper‑params
# ---------------------------------------------------------------------
LLAMA_MODEL_PATH = "sqlcoder-7b-2.Q4_K_M.gguf"
DB_URI           = "mysql+pymysql://root:admin@localhost/chatbot"

INDEX_PATH       = "schema_index/faiss_index.bin"
META_PATH        = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
TOP_K            = 3                        # top‑K from semantic search

N_CTX     = 4096
N_THREADS = 8

# ---------------------------------------------------------------------
# ─── Vector‑search helpers (ported from your 2nd script) ──────────────
# ---------------------------------------------------------------------
def load_faiss_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta

_fk_regex = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)

def build_reverse_fk_map(metadata: dict) -> dict[str, set[str]]:
    rev = {m["table_name"]: set() for m in metadata.values()}
    for m in metadata.values():
        for ref in _fk_regex.findall(m["create_stmt"]):
            if ref in rev:
                rev[ref].add(m["table_name"])
    return rev

def parse_forward_fks(ddl: str) -> set[str]:
    return set(_fk_regex.findall(ddl))

def semantic_search(q: str, embed_model, index, k: int):
    q_emb = np.array([embed_model.encode(q)], dtype="float32")
    _d, I = index.search(q_emb, k)
    return I[0]

def expand_with_related(idxs, meta, rev_fk):
    base = {meta[str(i)]["table_name"] for i in idxs}
    extra: Set[str] = set()
    for i in idxs:
        tbl = meta[str(i)]["table_name"]
        ddl = meta[str(i)]["create_stmt"]
        extra |= parse_forward_fks(ddl)
        extra |= rev_fk.get(tbl, set())
    return base | extra

# ---------------------------------------------------------------------
# ─── Thin wrapper around llama‑cpp for LangChain ─────────────────────
# ---------------------------------------------------------------------
def load_llama(path: str):
    print(f"🧠 Loading SQLCoder from {path}")
    return Llama(
        model_path=path,
        n_gpu_layers=0, n_ctx=N_CTX, n_threads=N_THREADS,
        n_batch=512, use_mlock=True, use_mmap=True, logits_all=False,
        verbose=True,
    )

class SQLCoderLLM(LLM):
    _model: Any = PrivateAttr()

    def __init__(self, model): super().__init__(); self._model = model
    @property
    def _llm_type(self): return "sqlcoder‑llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        out = self._model(prompt, max_tokens=512, stop=stop)
        text = out["choices"][0]["text"]
        # keep only the first statement
        if "SELECT" in text and ";" in text:
            text = text.split(";", 1)[0] + ";"
        return text.strip()

# ---------------------------------------------------------------------
# ─── App initialisation ──────────────────────────────────────────────
# ---------------------------------------------------------------------
print("🔧 Initialising FAISS / embedder / SQLCoder …")
_faiss, _meta = load_faiss_and_metadata(INDEX_PATH, META_PATH)
_rev_fk       = build_reverse_fk_map(_meta)
_embed_model  = SentenceTransformer(EMBED_MODEL_NAME)
_llama_raw    = load_llama(LLAMA_MODEL_PATH)
llm           = SQLCoderLLM(_llama_raw)
db            = SQLDatabase.from_uri(DB_URI)

# ---------------------------------------------------------------------
# ─── Monkey‑patch get_table_info so it obeys vector filtering ────────
# ---------------------------------------------------------------------
_last_question: str | None = None   # will be set right before each call

def _vector_filtered_schema(self, table_names: Optional[List[str]] = None) -> str:
    """
    If LangChain doesn’t pass table_names (it never does),
    run vector search on the *last question* to decide which tables
    to expose; else fall back to caller‑supplied list.
    """
    if table_names is None:
        if _last_question is None:
            raise ValueError("Question context missing for vector search.")
        idxs   = semantic_search(_last_question, _embed_model, _faiss, TOP_K)
        tables = expand_with_related(idxs, _meta, _rev_fk)
        table_names = list(tables)

    insp   = inspect(self._engine)
    exists = set(insp.get_table_names())
    schema_parts = []
    for t in table_names:
        if t not in exists: continue
        try:
            ddl = str(self._dialect.get_table_ddl(self._engine, t))
            schema_parts.append(f"CREATE TABLE {t} …\n{ddl}\n")
        except Exception:
            # fallback to columns only
            cols = insp.get_columns(t)
            col_defs = "\n  " + "\n  ".join(f"{c['name']} {c['type']}" for c in cols)
            schema_parts.append(f"CREATE TABLE {t} …{col_defs}\n")
    return "\n".join(schema_parts).strip()

# plug it in
db.get_table_info = _vector_filtered_schema.__get__(db)

# ---------------------------------------------------------------------
# ─── Build the LangChain agent ───────────────────────────────────────
# ---------------------------------------------------------------------
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

# ---------------------------------------------------------------------
# ─── Interactive loop ────────────────────────────────────────────────
# ---------------------------------------------------------------------
print("\n✨ Ready!  Ask me anything about your database.\n")
while True:
    try:
        question = input("❓> ").strip()
        if not question: continue
        if question.lower() in {"exit", "quit"}: break

        _last_question = question        # 👈 makes it available to schema patch
        response = agent.invoke({"input": question})

        print("\n📊 SQL executed:")
        print(response["intermediate_steps"][-1][1])    # last agent‑tool output
        print("\n✅ Answer:")
        print(response["output"])
    except KeyboardInterrupt:
        break
