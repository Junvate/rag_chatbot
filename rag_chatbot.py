from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from operator import itemgetter
import os  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import getpass
from typing import Literal
from typing_extensions import Annotated
from langchain_core.prompts import PromptTemplate

os.environ["TOGETHER_API_KEY"] = "sk-jpbyUPV7oHGME9IiE4cSbP4hoKsTHESPoZQM7R3FRgxwBfOl"

llm = ChatOpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="gpt-3.5-turbo",
)

embeddings = OpenAIEmbeddings(
    base_url="https://api2.aigcbest.top/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="text-embedding-3-large"
)



# 设置 LangSmith 追踪环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "rag"




# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")



class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# 添加边
graph_builder.add_edge(START, "analyze_query")  # 从 START 到 analyze_query
graph_builder.add_edge("analyze_query", "retrieve")  # 从 analyze_query 到 retrieve
graph_builder.add_edge("retrieve", "generate")  # 从 retrieve 到 generate
graph_builder.add_edge("generate", "__end__")  # 从 generate 到结束

graph = graph_builder.compile()


# ans=graph.invoke( {"question": "What does the end of the post say about Task Decomposition?"})
# print(ans["answer"])

template = "{question}"


custom_rag_prompt = PromptTemplate.from_template(template)

# 定义转换函数，将 Prompt 输出转换为 State 字典
def prompt_to_state(prompt_output):
    # prompt_output 是 StringPromptValue 对象，提取其文本内容
    question_text = prompt_output.text if hasattr(prompt_output, "text") else str(prompt_output)
    return {"question": question_text}

# 从 State 中提取 answer
def extract_answer(state):
    return state["generate"]["answer"]

parser = StrOutputParser()

chain = custom_rag_prompt | prompt_to_state | graph | extract_answer | parser



app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)




