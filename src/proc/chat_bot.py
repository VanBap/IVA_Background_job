import os
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# === DATA EMBEDDING + VECTOR DB ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPEN_API_KEY,base_url="https://models.inference.ai.azure.com")
vector_store = InMemoryVectorStore(embeddings)

# === Load and chuck contents of the blog ===
loader = WebBaseLoader(
    web_paths=("https://vi.wikipedia.org/wiki/T%E1%BA%ADp_%C4%91o%C3%A0n_Vingroup",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # class_=("post-content", "post-title", "post-header")
            class_=("mw-content-container")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
# print(f"all splits: {len(all_splits)}")
# print("=============================")

# === Index chunks ===
_ = vector_store.add_documents(documents=all_splits)

# === Define prompt for question-answering ===
# # C1
# prompt = hub.pull("rlm/rag-prompt")
# C2: customized
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# === Choose LLM model ===
llm = ChatOpenAI(model_name="gpt-4o",
                 api_key=OPEN_API_KEY,
                 base_url="https://models.inference.ai.azure.com",
                 temperature=0.0)

# === Define state for application ===
class State(TypedDict):
    question: str
    context: List[Document]
    answer:str

# === Define application steps (NODE) ===
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    # retrieved_docs = vector_store.similarity_search("question")
    print(f"retrieved_docs: {retrieved_docs}")
    print("======================================")
    return {"context": retrieved_docs}

def generate(state: State):

    # for doc in state["context"]:
    #     print(f"doc: {doc.page_content}")
    #     print("===================================================================")

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context":docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test (CONTROL FLOW)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "Vingroup là gì ?"})
# print(f"response: {response}")
print(f"Context: {response['context']}")
print(f"answer: {response['answer']}")
