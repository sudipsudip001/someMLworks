import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classification_node(state: State):
    """Classify the text into one of the categories: News, Blog, Research, or Other"""
    prompt = PromptTemplate.from_template(
        "Classify the following text into one of the categories: News, Blog, Research, or Other.\n\n"
        "Text: {text}\n\n"
        "Category:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    ''' Summarize the text in one short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

workflow = StateGraph(State)

workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction_node", entity_extraction_node)
workflow.add_node("summarization_node", summarization_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction_node")
workflow.add_edge("entity_extraction_node", "summarization_node")
workflow.add_edge("summarization_node", END)

app = workflow.compile()

def main():
    print("Hello from lang-classify!")
    sample_text = """OpenAI has recently announced the release of GPT-4o, a new iteration of their powerful language model. This model is expected to bring significant improvements in natural language understanding and generation capabilities. Researchers and developers are eager to explore the potential applications of GPT-4o in various fields, including healthcare, education, and customer service."""

    state_input = {"text": sample_text}
    result = app.invoke(state_input)

    print("Classifcation:", result["classification"])
    print("Entities:", result["entities"])
    print("Summary:", result["summary"])


if __name__ == "__main__":
    main()
