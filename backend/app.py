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
  # Receives JSON
    # {
    #     "input": {
    #         "messages": [
    #             {"role": "user","content": "question"}
    #         ]
    #     },
    #     "urls" : []
    # }
  data = request.get_json()  # Access the JSON data from the request body
  input = data["input"]
  urls = None
  for output in graph.stream(input):
    for key, value in output.items():
        # pprint.pprint(f"Output from node '{key}':")
        # pprint.pprint("---")
        # pprint.pprint(value, indent=2, width=80, depth=None)
        if key == "generate":
            urls = [doc.metadata["url"] for doc in value["documents"]]
        message = value["messages"][0].content
    pprint.pprint("\n---\n")
  data["input"]["messages"].append({"role": "assistant", "content": message})
  data["urls"].append(urls)
  return data
  
# LANGTRACE API

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
    
##################
# SELECT OUR MODEL
##################

# from langchain_community.chat_models import ChatOllama
# from langchain_experimental.llms.ollama_functions import OllamaFunctions

# local_llm = OllamaFunctions(model="llama3", streaming=True, temperature=0)
from langchain_openai import ChatOpenAI
local_llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", streaming=True)


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
# for filename in os.listdir("testdata"):
#   # Check if the file is a markdown file (ends with .md)
#   if filename.endswith(".md"):
#     # Construct the full path to the file
#     file_path = os.path.join("testdata", filename)
#     # Open the file in read mode
#     with open(file_path, "r") as f:
#     # Read the file content into a Document then append to doc_list array
#         doc = Document(page_content=f.read())
#         doc_list.append(doc)

# # Read medication documents json to an array of docs
import json, os
from langchain.docstore.document import Document
folder = "testdata/NHSmed"
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
    chunk_size=512, 
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
    print("Vector database loaded")
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

# # Create retriever tool
# from langchain.tools.retriever import create_retriever_tool

# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_medication_info",
#     "Search for information about medicines from guidance for patients",
# )

# tools = [retriever_tool]

# from langgraph.prebuilt import ToolExecutor

# tool_executor = ToolExecutor(tools)

##################
### AGENT STATE ##
##################
# We will define a graph.
# A state object passes around to each node.
# Our state will be a list of messages.
# Each node in our graph will append to it.

from typing import Annotated, Sequence, TypedDict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_question: BaseMessage # store, not append, the reworded question for access without indexing
    documents: List[Document] # same for retrieved documents
    # failed: int # default failed, updated by grader
        
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
    
def verify_question(state) -> Literal["rewrite", "reject"]:
    """
    Determines whether the initial question is relevant to capabilities of chatbot.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the question is relevant or not
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
        template="""You are a grader assessing relevance of the user question to the capabilities of this medical chatbot\n 
        Here is the user question: {question} \n
        If the question contains queries about medication or medication related information such as treatment options or side effects, then grade it as relevant\n
        Give a binary score 'yes' or 'no' score to indicate whether the question is relevant to medications""",
        input_variables=["question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    question = messages[-1]["content"] # most recent user query

    scored_result = chain.invoke({"question": question})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: QUESTION RELEVANT---")
        return "rewrite"

    else:
        print("---DECISION: QUESTION NOT RELEVANT---")
        return "reject"

############
### NODES ##
############

# print docs helper
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["rewrite_question"]
    messages = state["messages"]

    # Query Expansion Retrieval
    multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=local_llm
    )
    
    # Self Query + Filtered Search
    metadata_field_info = [
        AttributeInfo(
            name="med_name",
            description="Name of the medication. If a common brand name exists it may be in brackets after",
            type="string"
        ),
        AttributeInfo(
            name="document_description", 
            description="The specific topics about the medication",
            type="string"
        ),
        AttributeInfo(
            name="page_description", 
            description="What conditions the medication is used to treat. May contain alternated brand names",
            type="string"
        ),
    ]
    document_content_description = "Information about a specific medication"
    self_retriever = SelfQueryRetriever.from_llm(
        local_llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
    )
    
    # Run Retrievers
    multi_documents = multi_retriever.invoke(question.content)
    print(f"\n Query Expansion: {len(multi_documents)} Documents Returned \n")
    #pretty_print_docs(multi_documents)
    self_documents = self_retriever.invoke(question.content)
    print(f"\n Self Query + Filter: {len(self_documents)} Documents Returned \n")
    #pretty_print_docs(self_documents)
    
    # Combine docs and deduplicate
    documents = []
    metadata_set = set()
    for doc in multi_documents + self_documents:
        if doc.page_content not in metadata_set:
            metadata_set.add(doc.page_content)
            documents.append(doc)
    print(f"\n UNIQUE DOCUMENTS: {len(metadata_set)} \n")
    return {"documents": documents, "rewrite_question": question, "messages": messages}

def reject(state):
    """
    Reject the question, offer a list of capabilities and signpost to key resources

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the reject response appended to messages
    """
    print("---REJECT QUERY---")
    question = state["rewrite_question"]
    if question == None: # if rejected immediately then get original question
        messages = state["messages"]
        question = messages[-1]["content"] 

    msg = [
        HumanMessage(
            content=f""" \n 
    Reject the following question politely. Inform the user that you are a chatbot only capable of answering questions using official NHS guidance on medications, their side effects, interactions, dosage, administration, lifestyle considerations, efficacy and monitoring. \n
    Here is the question:
    \n ------- \n
    {question} 
    \n ------- \n
    Use the question to give specific reasons why you are unable to answer. \n
    Always provide this link to the NHS website so they can search for relevant information theirself https://www.nhs.uk/medicines/. \n
    If the question is asking for specific medical advice, advise them to speak to a healthcare professional, call 111 for immediate advice or call 999 in an emergency.
    """,
        )
    ]

    model = local_llm
    response = model.invoke(msg)
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
    question = messages[-1]["content"] # latest user query

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the question and try to reason about the underlying semantic intent / meaning. Pay particular attention to any key medical terms \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Correct any spelling errors in medication names and formulate an improved question for searching medical information.
    Only respond with the reworded question, with no other preamble or conversation: """,
        )
    ]

    model = local_llm
    response = model.invoke(msg)
    print(response)
    return {
        "messages": [response], 
        "rewrite_question" : response, # access from state without indexing
    }
    
    
# def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """
#     # class for output
#     class GradeDocuments(BaseModel):
#         """Binary score for relevance check on retrieved documents."""

#         binary_score: str = Field(
#             description="Documents are relevant to the question, 'yes' or 'no'"
#         )

#     # LLM with function call
#     llm = local_llm
#     structured_llm_grader = llm.with_structured_output(GradeDocuments)

#     # Prompt
#     system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#         If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#         ]
#     )
    
#     retrieval_grader = grade_prompt | structured_llm_grader

#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    
#     question = state["rewrite_question"]
#     documents = state["documents"]
#     failed = state["failed"]
#     messages = state["messages"]

#     # Score each doc
#     filtered_docs = []
#     for d in documents:
#         score = retrieval_grader.invoke(
#             {"question": question, "document": d.page_content}
#         )
#         grade = score.binary_score
#         if grade == "yes":
#             print("---GRADE: DOCUMENT RELEVANT---")
#             print(d)
#             filtered_docs.append(d)
#             failed = 0 # mark as not failed if ANY doc is relevant
#         else:
#             print("---GRADE: DOCUMENT NOT RELEVANT---")
#             print(d)
#             continue
#     return {"documents": filtered_docs, "rewrite_question": question, "failed": failed, "messages": messages}

from langchain_community.document_compressors import FlashrankRerank

def rank_documents(state):
    """
    Reranks the documents according to relevance to original question 

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only top relevant documents
    """
    
    print("---RANKING DOCUMENT RELEVANCE TO QUESTION---")
    
    question = state["rewrite_question"]
    documents = state["documents"]
    messages = state["messages"]
    # Use FlashrankRerank to rank and return top n relevant documents
    reranker = FlashrankRerank(top_n=4)
    reranked_docs = reranker.compress_documents(documents=documents, query=question.content) 

    print(f"\n Returning top {len(reranked_docs)} out of {len(documents)} Documents \n")

    return {"documents": reranked_docs, "rewrite_question": question, "messages": messages}


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
    docs = state["documents"]
    question = state["rewrite_question"]
    
    doctext = ""
    for doc in docs:
        doctext += f"{doc.page_content}\n\n"
    
    pretty_print_docs(docs)
    
    # Prompt 
    prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an assistant for answering questions about medications.
         Rely heavily on the following pieces of retrieved context to answer the question.
         In your response do not use the phrase "in the provided context", instead say "on the NHS website"
         If you don't know the answer, just say that you are unable to find any specific information from the NHS Medicines website, but offer adjacent relevant advice from the provided context.
         Question: {question}
         Context: {context}
         Answer: """),])

    # LLM
    llm = local_llm

    # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm #| StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": doctext, "question": question})
    return {"messages": [response],"documents": docs, "rewrite_question": question,}


#############
### GRAPH ###
#############

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("rewrite", rewrite)
workflow.add_node("reject", reject)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rank_documents", rank_documents)
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant

# Verify first then reject or rewrite for retrieval
workflow.set_conditional_entry_point(
    verify_question,
    {
        "rewrite": "rewrite",
        "reject": "reject",
    },
)

# Then send to retrieve
workflow.add_edge("rewrite", "retrieve")

# Grade retrieved documents
workflow.add_edge("retrieve", "rank_documents")

# # Edges taken after grade to decide to reject or generate
# workflow.add_conditional_edges(
#     "rank_documents",
#     # Assess agent decision
#     decide_to_generate,
# )
workflow.add_edge("rank_documents", "generate")

# Finish after generate
workflow.add_edge("generate", END)

# Finish after reject
workflow.add_edge("reject", END)

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
    