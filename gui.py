import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLineEdit, QLabel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QDesktopWidget
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal
from PyQt5.QtWidgets import QListWidget
import os
from langchain.tools import tool
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
from PyQt5.QtWidgets import QHBoxLayout
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
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel , Field


# 设置环境变量
os.environ["TOGETHER_API_KEY"] = "sk-jpbyUPV7oHGME9IiE4cSbP4hoKsTHESPoZQM7R3FRgxwBfOl"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-wooden-organisation-94"

# 初始化 LLM 和 Embeddings
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

# PyQt 应用程序
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.vector_store = None
        self.graph = None
        self.initUI()
        self.center()  # 初始化时将窗口居中

    def initUI(self):
        self.setWindowTitle('LangChain Q&A')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # 网址输入部分（改为多栏）
        self.urls_label = QLabel('URLs:')
        layout.addWidget(self.urls_label)
        self.urls_list = QListWidget()
        layout.addWidget(self.urls_list)
        urls_buttons_layout = QHBoxLayout()
        self.add_url_button = QPushButton('添加 URL')
        self.add_url_button.clicked.connect(self.add_url)
        urls_buttons_layout.addWidget(self.add_url_button)
        self.remove_url_button = QPushButton('删除选中 URL')
        self.remove_url_button.clicked.connect(self.remove_url)
        urls_buttons_layout.addWidget(self.remove_url_button)
        layout.addLayout(urls_buttons_layout)

        # 问题输入
        self.question_label = QLabel('问题:')
        layout.addWidget(self.question_label)
        self.question_input = QLineEdit()
        layout.addWidget(self.question_input)

        # 提交按钮
        self.submit_button = QPushButton('提交')
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        # 状态显示（用于“加载中...”）
        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        # 回答输出
        self.answer_label = QLabel('回答:')
        layout.addWidget(self.answer_label)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.setLayout(layout)

    def center(self):
        """将窗口移动到屏幕中心"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def add_url(self):
        """添加新的 URL"""
        url, ok = QInputDialog.getText(self, '添加 URL', '输入 URL:')
        if ok and url:
            self.urls_list.addItem(url)

    def remove_url(self):
        """删除选中的 URL"""
        for item in self.urls_list.selectedItems():
            self.urls_list.takeItem(self.urls_list.row(item))

    def on_submit(self):
        global vector_store
        urls = [self.urls_list.item(i).text() for i in range(self.urls_list.count())]
        question = self.question_input.text()

        if not urls or not question:
            self.answer_output.setText("请输入 URL 和问题。")
            return

        # 显示“加载中...”
        self.status_label.setText("加载中...")
        QApplication.processEvents()  # 强制刷新 UI

        # 加载和处理文档
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(doc_splits)

        # 加载完成后清空状态
        self.status_label.setText("")

        # 构建 LangGraph 图（保持不变）
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", ToolNode([retrieve]))
        graph_builder.add_node("generate", generate)
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        self.graph = graph_builder.compile()

        # 准备初始状态并执行图
        state = {"messages": [HumanMessage(content=question)]}
        result = self.graph.invoke(state)

        # 获取回答
        answer = None
        for message in result["messages"]:
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = message.content
                break

        self.answer_output.setText(answer if answer else "未生成回答。")



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
        
# LLM with function call
llm_grader = ChatOpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="gpt-4o-mini",
)

structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


# 定义 retrieve、query_or_respond 和 generate 函数（保持不变）
@tool(response_format="content_and_artifact")
def retrieve(question: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(question, k=4)
    docs_to_use = []
    for doc in retrieved_docs:
        res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if res.binary_score == "yes":
            docs_to_use.append(doc)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in docs_to_use
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())