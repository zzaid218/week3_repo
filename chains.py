import os
from dotenv import load_dotenv

load_dotenv()  

from flask import Flask, render_template, request
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# LLM
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o-mini"
)

# Chain 1 — Summarization
summary_prompt = PromptTemplate.from_template(
    "Summarize the following tasks briefly:\n{text}"
)
summary_chain = summary_prompt | llm

# Chain 2 — Word Count
wordcount_prompt = PromptTemplate.from_template(
    "Count the number of words in this text:\n{summary}"
)
wordcount_chain = wordcount_prompt | llm

# Chain 3 — Classification
classification_prompt = PromptTemplate.from_template(
    "Classify this text into one topic (Work, Personal, Education, Business, Health):\n{text}"
)
classification_chain = classification_prompt | llm

# Chain 4 — Task Planner
task_prompt = PromptTemplate.from_template("""
You are an AI task planner.

Tasks:
{tasks}

Generate a single output including:

1. Summary of Tasks:
   - Write a clear and descriptive summary of all tasks.
   - At the end of the summary, add a line exactly like this:
     "Word Count of Summary: X"
   - Replace X with the number of words in the summary.
   - Also add a short description:
     "This number represents the total words in the summary, covering all tasks in brief."

2. Categories: List each task under Work or Personal.

3. Priority Levels: Assign High, Medium, Low for each task.

Example format:

Summary
[Your summary text here]
Word Count of Summary: 29
(This number represents the total words in the summary)

Categories
- Work → task1, task2
- Personal → task3

Priority Levels
- High → task1
- Medium → task2
- Low → task3

Ensure the word count line is always included exactly as shown above.
""")
task_chain = task_prompt | llm

# Flask Route
@app.route("/", methods=["GET", "POST"])
def index():
    summary = word_count = category = result = None
    result = None
    tasks = ""

    if request.method == "POST":
        tasks = request.form["tasks"]

        # Only generate final plan (summary, categories, priorities inside it)
        result = task_chain.invoke({"tasks": tasks}).content

    return render_template(
        "index.html",
        tasks=tasks,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)