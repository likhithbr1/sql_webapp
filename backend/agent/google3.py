from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def build_agent_chain(sqlcoder_llm):
    query_chain = create_sql_agent(
        llm=sqlcoder_llm,
        db=SQLDatabase.from_uri(DB_URI),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # ✅ Directly pass {"question": "..."} to custom_schema_chain
    table_chain = RunnableLambda(custom_schema_chain)

    full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    return full_chain
