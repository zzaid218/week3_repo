import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Use the classic package for these specific imports
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Define the Tool
@tool
def count_word_frequency(text: str, word: str) -> int:
    """Counts the exact number of times a specific word appears in the provided text."""
    words = text.lower().split()
    return words.count(word.lower())

# 3. Setup Tools & LLM
tools = [count_word_frequency]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Create the Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert text analyst. Use your tools to count words accurately."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 5. Create the Agent & Executor
# Note: Ensure the 'input' key matches what you pass to invoke()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Run it
my_text = "Cybersecurity is important. This cybersecurity training is in week three."
query = f"How many times does the word 'cybersecurity' appear in this text: {my_text}"

print("\n--- AGENT STARTING ---")
response = agent_executor.invoke({"input": query})

print("\n--- FINAL ANSWER ---")
print(response["output"])