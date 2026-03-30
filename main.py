import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 🤖 2. Initialize Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🧠 3. Define the Prompt
summary_prompt = PromptTemplate.from_template(
    """You are a precise summarization assistant.
    Summarize the text below concisely without repeating ideas.

    TEXT:
    {text}

    SUMMARY:"""
)

# 🔗 4. Create the Chain (The LCEL way)
# This pipes the prompt into the model, then parses the output into a clean string
chain = summary_prompt | llm | StrOutputParser()

# 🚀 5. Run the Chain
text_to_summarize = """
Artificial intelligence is rapidly changing the world. 
It helps automate tasks and improves decision making. 
Many industries like healthcare, finance, and education are adopting AI technologies. 
However, there are also concerns about job loss and ethical issues.
"""

summary = chain.invoke({"text": text_to_summarize})

print("\n========== FINAL SUMMARY ==========\n")
print(summary)