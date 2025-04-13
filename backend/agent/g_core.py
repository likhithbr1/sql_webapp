


from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnableMap

def build_agent_chain(sqlcoder_llm):
    query_chain = create_sql_agent(
        llm=sqlcoder_llm,
        db=SQLDatabase.from_uri(DB_URI),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    table_chain = RunnableMap({"input": itemgetter("question")}) | RunnableLambda(custom_schema_chain)

    full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    return full_chain













import json
import re
import numpy as np
import faiss
import torch
import sqlparse

from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------ Configuration ------------------------------
INDEX_PATH = "schema_index/faiss_index.bin"
META_PATH = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
TOP_K = 3
DB_URI = "sqlite:///chatbot.db"
MODEL_NAME = "defog/sqlcoder-7b-2"

# ------------------------------ FAISS + Metadata ------------------------------
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
    return "\n\n".join(
        meta["create_stmt"] for meta in metadata.values() if meta["table_name"] in table_names
    )

def get_relevant_schema_snippet(query: str, embed_model, faiss_index, metadata, rev_fk_map, top_k: int) -> str:
    idxs = semantic_search(query, embed_model, faiss_index, top_k)
    final_tables = expand_with_related(idxs, metadata, rev_fk_map)
    return build_schema_snippet(final_tables, metadata)

# ------------------------------ SQLCoder LLM Wrapper ------------------------------
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Mapping, Any

class SQLCoderLLM(LLM):
    def _llm_type(self) -> str:
        return "sqlcoder"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": MODEL_NAME}

# ------------------------------ Query Generation ------------------------------
def generate_query(prompt: str) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    result = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return sqlparse.format(result.split("[SQL]")[-1], reindent=True)

# ------------------------------ Initialization ------------------------------
def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model, _tokenizer, _model, _engine, sqlcoder_llm

    print("🔧 Initializing vector index, embeddings, and model…")
    _faiss_index, _metadata = load_faiss_and_metadata(INDEX_PATH, META_PATH)
    _rev_fk_map = build_reverse_fk_map(_metadata)
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
        if vram > 15e9:
            print("📦 Loading model in fp16 (high VRAM)")
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            print("📦 Loading model in 8-bit mode (low VRAM)")
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto"
            )
    else:
        print("⚙️ Using CPU for model inference")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        _model.to("cpu")

    _engine = create_engine(DB_URI)
    sqlcoder_llm = SQLCoderLLM()
    print("✅ All models loaded and DB connected.")
    return sqlcoder_llm

# ------------------------------ LangChain Integration ------------------------------
from operator import itemgetter
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough

def custom_schema_chain(input_dict: dict) -> str:
    return get_relevant_schema_snippet(
        input_dict["question"], _embed_model, _faiss_index, _metadata, _rev_fk_map, TOP_K
    )

def build_agent_chain(sqlcoder_llm) -> any:
    query_chain = create_sql_agent(
        llm=sqlcoder_llm,
        db=SQLDatabase.from_uri(DB_URI),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    table_chain = {"input": itemgetter("question")} | custom_schema_chain
    full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    return full_chain

# ------------------------------ Process Input Question ------------------------------
def process_question_agentic(question: str) -> dict:
    try:
        agent_chain = build_agent_chain(sqlcoder_llm)
        result = agent_chain.invoke({"question": question})
        return result
    except Exception as e:
        return {"sql": None, "results": [], "error": str(e)}

# ------------------------------ Main Usage ------------------------------
if __name__ == "__main__":
    sqlcoder_llm = init_models()
    user_question = "What are the top 5 orders by revenue in 2022?"
    answer = process_question_agentic(user_question)

    print("Generated SQL:\n", answer.get("sql"))
    print("Query Results:\n", answer.get("results"))
    if "error" in answer:
        print("Error:", answer.get("error"))

