# ---------------------------------------------------------------------
# 3 bis.  Build a strict prompt so the agent *must* fetch schema first
# ---------------------------------------------------------------------
PREFIX = """You are an expert SQL analyst.
**Always start by calling the tool `sql_db_schema`** with the user question
as its input.  Only after you have examined the returned CREATE TABLE
statements should you think about which columns to query.
Follow this tool‑calling format exactly:

Thought: you must think about what to do
Action: the_tool_name
Action Input: the input for the tool
Observation: the result of the tool
(Repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the result for the user
"""

SUFFIX = """Begin!  Remember: start with `sql_db_schema`."""


# ---------------------------------------------------------------------
# 4.  Collect tools (same as before) & build the agent
# ---------------------------------------------------------------------
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit

std_db_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
std_db_tools = [t for t in std_db_tools if t.name != "sql_db_schema"]  # drop default
tools = [vector_schema_tool, *std_db_tools]

from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX},
)


# ---------------------------------------------------------------------
# 5.  Interactive loop   (switch to .invoke() to silence the warning)
# ---------------------------------------------------------------------
print("\n✨ Ready!  Ask me anything about your database.\n")
while True:
    try:
        question = input("❓> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = agent.invoke({"input": question})   # ← no deprecation warning

        # Debug trace
        for act, obs in result["intermediate_steps"]:
            print(f"• {act.tool}: {act.tool_input}")

        print("\n✅ Final answer:")
        print(result["output"])

    except KeyboardInterrupt:
        break
