import json
import re
import numpy as np
import faiss
import torch
import sqlparse

from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INDEX_PATH = "schema_index/faiss_index.bin"
META_PATH = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
TOP_K = 3
DB_URI = "sqlite:///chatbot.db"  # SQLite DB inside Colab
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

# ----------------------------------------------------------------------
# FAISS + Schema Handling
# ----------------------------------------------------------------------

def load_faiss_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta

def build_reverse_fk_map(metadata: dict) -> dict:
    rev_map = {m["table_name"]: set() for m in metadata.values()}
    fk_pattern = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)
    for m in metadata.values():
        for ref in fk_pattern.findall(m["create_stmt"]):
            if ref in rev_map:
                rev_map[ref].add(m["table_name"])
    return rev_map

def parse_forward_fks(ddl: str) -> set[str]:
    return set(re.findall(r"REFERENCES\s+`?(\w+)`?", ddl, re.IGNORECASE))

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

# ----------------------------------------------------------------------
# Hugging Face SQLCoder Inference
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Initialization + Inference
# ----------------------------------------------------------------------

def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model, _tokenizer, _model

    print("🔧 Loading models and resources...")
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


def retry_generate_query(question: str, schema: str, max_retries: int = 2) -> dict:
    attempt = 0
    prompt = PROMPT_TEMPLATE.format(question=question, schema=schema)

    while attempt <= max_retries:
        try:
            final_sql = generate_query(prompt).strip()

            # Execute SQL
            engine = create_engine(DB_URI)
            with engine.connect() as connection:
                result = connection.execute(text(final_sql))
                rows = [dict(row._mapping) for row in result.fetchall()]
                return {
                    "sql": final_sql,
                    "results": rows
                }

        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return {
                    "sql": final_sql,
                    "results": [],
                    "error": f"Failed after {max_retries} retries. Last error: {str(e)}"
                }

            # 🔁 Smart retry prompt with feedback loop
            prompt = f"""### Original Question:
{question}

### Previous SQL:
{final_sql}

### Error Message:
{str(e)}

### Database Schema:
{schema}

### Revised SQL:
[SQL]"""




def process_question(question: str) -> dict:
    try:
        # Step 1: Table selection via vector search
        idxs = semantic_search(question, _embed_model, _faiss_index, TOP_K)
        final_tables = expand_with_related(idxs, _metadata, _rev_fk_map)
        schema = build_schema_snippet(final_tables, _metadata)

        # Step 2: Build prompt and try to generate + execute query
        prompt = PROMPT_TEMPLATE.format(question=question, schema=schema)
        final_sql = generate_query(prompt).strip()

        engine = create_engine(DB_URI)
        with engine.connect() as connection:
            result = connection.execute(text(final_sql))
            rows = [dict(row._mapping) for row in result.fetchall()]

        return {
            "sql": final_sql,
            "results": rows
        }

    except Exception as e:
        print("⚠️ Initial execution failed. Retrying with feedback...")
        return retry_generate_query(question, schema, max_retries=2)

