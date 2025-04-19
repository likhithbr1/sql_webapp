# ---------------------------------------------------------------------
# ─── 1.  Build SQLCoder LLM + DB  (unchanged)  ───────────────────────
# ---------------------------------------------------------------------
print("🔧 Initialising FAISS / embedder / SQLCoder …")
_faiss, _meta = load_faiss_and_metadata(INDEX_PATH, META_PATH)
_rev_fk       = build_reverse_fk_map(_meta)
_embed_model  = SentenceTransformer(EMBED_MODEL_NAME)
_llama_raw    = load_llama(LLAMA_MODEL_PATH)
llm           = SQLCoderLLM(_llama_raw)
db            = SQLDatabase.from_uri(DB_URI)

# ---------------------------------------------------------------------
# ─── 2.  Custom FAISS‑backed schema tool  ────────────────────────────
# ---------------------------------------------------------------------
from langchain.tools import Tool   # ← generic tool wrapper

def _schema_from_vector(question: str) -> str:
    """
    ① embed the question  ② FAISS top‑K ③ FK expansion ④ build DDL snippet
    """
    idxs   = semantic_search(question, _embed_model, _faiss, TOP_K)
    tables = expand_with_related(idxs, _meta, _rev_fk)
    print(f"\n🧩 Tables selected for prompt: {sorted(tables)}\n")
    return build_schema_snippet(tables, _meta)

vector_schema_tool = Tool(
    name="sql_db_schema",            # ⚠️ same name as the default → overrides it
    func=_schema_from_vector,
    description=(
        "Given a natural‑language question, return ONLY the schema DDL for the "
        "relevant tables as Markdown. Uses semantic search over table DDLs."
    ),
)

# ---------------------------------------------------------------------
# ─── 3.  Collect standard DB tools, but drop the old schema tool  ────
# ---------------------------------------------------------------------
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit

std_db_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
std_db_tools = [t for t in std_db_tools if t.name != "sql_db_schema"]  # remove default

tools = [vector_schema_tool, *std_db_tools]

# ---------------------------------------------------------------------
# ─── 4.  Build the agent with initialise_agent()  ────────────────────
# ---------------------------------------------------------------------
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,          # so we can inspect what happened
    handle_parsing_errors=True,
)

# ---------------------------------------------------------------------
# ─── 5.  Interactive loop  ───────────────────────────────────────────
# ---------------------------------------------------------------------
print("\n✨ Ready!  Ask me anything about your database.\n")
while True:
    try:
        question = input("❓> ").strip()
        if not question: 
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = agent({"input": question})

        # ---------------- debug / inspect ----------------
        steps = result.get("intermediate_steps", [])
        if steps:
            print("\n🪜 Agent reasoning trace:")
            for action, obs in steps:              # type: ignore
                print(f"• {action.tool}: {action.tool_input}")
        # -------------------------------------------------

        print("\n✅ Final answer:")
        print(result["output"])

    except KeyboardInterrupt:
        break
