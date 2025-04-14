import json
import re
import numpy as np
import faiss
import torch
import sqlparse
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Mapping, Any
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# ------------------------------
# Configuration
# ------------------------------
INDEX_PATH = "schema_index/faiss_index.bin"
META_PATH = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
TOP_K = 3
DB_URI = "sqlite:///chatbot.db"
MODEL_NAME = "defog/sqlcoder-7b-2"

PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
The query will run on a database with the following schema:
{schema}

### SQL Query
[SQL]
"""

# ------------------------------
# FAISS + Schema Handling
# ------------------------------
def load_faiss_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta

def build_reverse_fk_map(metadata: dict) -> dict:
    rev_map = {m["table_name"]: set() for m in metadata.values()}
    fk_pattern = re.compile(r"REFERENCES\s+?(\w+)?", re.IGNORECASE)
    for m in metadata.values():
        for ref in fk_pattern.findall(m["create_stmt"]):
            if ref in rev_map:
                rev_map[ref].add(m["table_name"])
    return rev_map

def parse_forward_fks(ddl: str) -> set[str]:
    return set(re.findall(r"REFERENCES\s+?(\w+)?", ddl, re.IGNORECASE))

def semantic_search(query: str, embed_model, faiss_index, top_k: int):
    q_emb = embed_model.encode(query)
    q_emb = np.array([q_emb], dtype="float32")
    _, I = faiss_index.search(q_emb, top_k)
    return I[0]

def expand_with_related(idx_list, metadata, rev_fk_map):
    tables = {metadata[str(i)]["table_name"] for i in idx_list}
    extra = set()
    for i in idx_list:
        ddl = metadata[str(i)]["create_stmt"]
        table = metadata[str(i)]["table_name"]
        extra.update(parse_forward_fks(ddl))
        extra.update(rev_fk_map.get(table, set()))
    return tables.union(extra)

def build_schema_snippet(table_names: set[str], metadata: dict) -> str:
    return "\n\n".join(
        m["create_stmt"] for m in metadata.values()
        if m["table_name"] in table_names
    )

# ------------------------------
# Hugging Face SQLCoder Inference
# ------------------------------
def generate_query(prompt: str) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = _model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    outputs = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)

# ------------------------------
# Initialization + Inference
# ------------------------------
def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model, _tokenizer, _model

    print("\U0001F527 Loading models and resources...")
    _faiss_index, _metadata = load_faiss_and_metadata(INDEX_PATH, META_PATH)
    _rev_fk_map = build_reverse_fk_map(_metadata)
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vram = torch.cuda.get_device_properties(0).total_memory

    if vram > 15e9:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto"
        )
    print("✅ Model loaded (VRAM:", round(vram / 1e9, 2), "GB)")

# ------------------------------
# LangChain SQL Agent Integration
# ------------------------------
class SQLCoderLLM(LLM):
    def _llm_type(self) -> str:
        return "sqlcoder"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return generate_query(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": MODEL_NAME}

def custom_schema_chain(input_dict: dict) -> str:
    question = input_dict["question"]
    idxs = semantic_search(question, _embed_model, _faiss_index, TOP_K)
    tables = expand_with_related(idxs, _metadata, _rev_fk_map)
    return build_schema_snippet(tables, _metadata)

def build_agent_chain():
    sqlcoder_llm = SQLCoderLLM()
    db = SQLDatabase.from_uri(DB_URI)
    query_chain = create_sql_agent(
        llm=sqlcoder_llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    table_chain = {"input": itemgetter("question")} | custom_schema_chain
    return RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

def process_question_agentic(question: str) -> dict:
    try:
        agent_chain = build_agent_chain()
        result = agent_chain.invoke({"question": question})
        return result
    except Exception as e:
        return {"sql": None, "results": [], "error": str(e)}

if __name__ == "__main__":
    init_models()
    question = "List top 5 customers by invoice total"
    output = process_question_agentic(question)
    print("Generated SQL:\n", output.get("sql"))
    print("Results:\n", output.get("results"))
    if "error" in output:
        print("Error:", output.get("error"))
