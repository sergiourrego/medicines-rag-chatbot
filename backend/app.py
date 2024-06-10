############
## FLASK ###
############

from flask import Flask, request
app = Flask(__name__)

# POST method for chat messages
from langchain.load.dump import dumps
import pprint

@app.route('/messages', methods=['POST'])
def simple_message():
  # JSON of whole conversation
    # {
    #     "messages": [
    #         {role: user, content: message},
    #     ]
    # }
  data = request.get_json()  # Access the JSON data from the request body
  inputs = data
  for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
        # LangChain returns custom AIMessage object, parse with dumps
        message = value["messages"][0].content
    pprint.pprint("\n---\n")
  data["messages"].append({"role": "assistant", "content": message})
  return data
  
###################
#### LANGCHAIN ####
###################
import os, sys

# Set our LangChain API
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# Check for API Key
if "LANGCHAIN_API_KEY" in os.environ:
    LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
else:
    print("LANGCHAIN API Key missing from .env")
    sys.exit

# Select our model
# from langchain_community.chat_models import ChatOllama
# from langchain_experimental.llms.ollama_functions import OllamaFunctions

# local_llm = OllamaFunctions(model="llama3", streaming=True, temperature=0)
from langchain_openai import ChatOpenAI
local_llm = ChatOpenAI(temperature=0.3, model="gpt-4-turbo", streaming=True)


########################
# CREATE VECTOR DATABASE
########################

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# # Firecrawl Scrape
# from langchain_community.document_loaders import FireCrawlLoader
# urls = []
# docs = [FireCrawlLoader(api_key="xxxxx", url=url, mode="scrape").load() for url in urls]

# TEST: read md files
# Loop through all files in the directory
# doc_list = []
# for filename in os.listdir("backend/testdata"):
#   # Check if the file is a markdown file (ends with .md)
#   if filename.endswith(".md"):
#     # Construct the full path to the file
#     file_path = os.path.join("backend/testdata", filename)
#     # Open the file in read mode
#     with open(file_path, "r") as f:
#     # Read the file content into a Document then append to doc_list array
#         doc = Document(page_content=f.read())
#         doc_list.append(doc)

# # Read medication documents json to an array of docs
import json, os
from langchain.docstore.document import Document

folder = "backend/testdata/NHSmed"
doc_list = []
for filename in os.listdir(folder):
  # Check if file is a JSON file
  if filename.endswith(".json") and filename != "medication_table.json":
    file_path = os.path.join(folder, filename)
    try:
      with open(file_path, 'r') as json_file:
        dict = json.load(json_file)
        for obj in dict.values():
          obj = json.loads(obj)
          doc = Document(**obj)
          doc_list.append(doc)
      print(f"Loaded Medication: {filename}")  # Print success message with filename
    except FileNotFoundError:
      print(f"Error: Medication file not found: {filename}")  # Print error message
    except json.JSONDecodeError:
      print(f"Error: Invalid JSON format in medication file: {filename}")  # Print error message for invalid JSON


#Split docs into chunks using tiktoken
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #EXPERIMENT WITH CHUNK SIZE
    chunk_size=128, 
    chunk_overlap=64,
)
doc_splits = text_splitter.split_documents(doc_list)
print("Split Documents")

# # Filter out complex metadata and ensure proper document formatting
# from langchain_community.vectorstores.utils import filter_complex_metadata

# filtered_docs = []
# for doc in doc_splits:
#     if isinstance(doc, Document) and hasattr(doc, 'metadata'):
#         clean_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
#         filtered_docs.append(Document(page_content=doc.page_content, metadata=clean_metadata))

# Configure embedder
# from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# import torch
# print(torch.cuda.is_available())

model_name = "Alibaba-NLP/gte-large-en-v1.5"
model_kwargs = {'device': 'cuda', "trust_remote_code": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

# if vectorDB exists load it
# feat: rebuild vectorDB on source file change
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", collection_name="rag-chroma", embedding_function=hf,)
else:
#Add to vectorDB, with persistent file
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=hf,
        persist_directory="./chroma_db",
    )
    # persist db
    vectorstore.persist()
    print("Vector database created")

###########
# RETRIEVER
###########

retriever = vectorstore.as_retriever()

# Create retriever tool
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_medication_info",
    "Search for information about medicines from guidance for patients",
)

tools = [retriever_tool]

from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)

##################
### AGENT STATE ##
##################
# We will define a graph.
# A state object passes around to each node.
# Our state will be a list of messages.
# Each node in our graph will append to it.

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


#############
### EDGES####
#############


def grade_documents(state) -> Literal["generate", "agent"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = local_llm

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "agent"

############
### NODES ##
############


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = local_llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # find latest question
    question = messages[-1]["content"]

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. Pay particular attention to any key medical terms \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question to search medical documents: """,
        )
    ]

    # Grader
    model = local_llm
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # Prompt 
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an assistant for answering questions about medications.
         Use the following pieces of retrieved context to answer the question.
         If you don't know the answer, just say that you are unable to answer with the information available.
         Question: {question}
         Context: {context}
         Answer: """),])

    # LLM
    llm = local_llm

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm #| StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


#############
### GRAPH ###
#############

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant

# workflow.add_node("second_agent", agent) # second loop after rewrite
# workflow.add_node("second_retrieve", retrieve)  # second relevance check exits on fail to stop looping


# Call rewrite initially to ensure best search outcome
workflow.set_entry_point("rewrite")

# Then send to agent to decide whether to retrieve
workflow.add_edge("rewrite", "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)


# Edges taken after the retrieve is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
# Finish after generate
workflow.add_edge("generate", END)


# # Second loop
# workflow.add_conditional_edges(
#     "second_agent",
#     # Assess agent decision
#     tools_condition,
#     {
#         # Translate the condition outputs to nodes in our graph
#         "tools": "second_retrieve",
#         END: END,
#     },
# )
# # Exits instead of rewriting again
# workflow.add_conditional_edges(
#     "second_retrieve",
#     # Assess agent decision
#     grade_documents,
#     {
#     # Translate the condition outputs to nodes in our graph
#     "generate": "generate",
#     "rewrite": END,
#     },
# )

# Compile
graph = workflow.compile()

# ##TEST RUN###
# import pprint
# print(vectorstore)
# inputs = {
#     "messages": [
#         ("user", "Will Ciprofloxacin interact with my hibiscus eye drops. I heard it can"),
#     ]
# }
# lastmessage = ""
# for output in graph.stream(inputs):
#     for key, value in output.items():
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint("---")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#         message_content = value['messages'][0].content
#     pprint.pprint("\n---\n")
# print(message_content)
    
# Run Flask
if __name__ == '__main__':
  app.run(debug=False)
    