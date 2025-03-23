import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLineEdit, QLabel
from PyQt5.QtWidgets import QInputDialog, QDesktopWidget, QListWidget, QHBoxLayout
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import List
import os
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
)
import os  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from typing_extensions import List

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
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

open_ai_model="gpt-3.5-turbo"
# 初始化 LLM 和 Embeddings
llm = ChatOpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model=open_ai_model,
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
        self.center()

    def initUI(self):
        self.setWindowTitle('LangChain Q&A')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # URL 输入部分
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

        # 状态显示
        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        # 回答输出
        self.answer_label = QLabel('回答:')
        layout.addWidget(self.answer_label)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        # 参考文档段落输出
        self.references_label = QLabel('参考文档段落:')
        layout.addWidget(self.references_label)
        self.references_output = QTextEdit()
        self.references_output.setReadOnly(True)
        layout.addWidget(self.references_output)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def add_url(self):
        url, ok = QInputDialog.getText(self, '添加 URL', '输入 URL:')
        if ok and url:
            self.urls_list.addItem(url)

    def remove_url(self):
        for item in self.urls_list.selectedItems():
            self.urls_list.takeItem(self.urls_list.row(item))

    def on_submit(self):
        global vector_store
        urls = [self.urls_list.item(i).text() for i in range(self.urls_list.count())]
        question = self.question_input.text()

        if not urls or not question:
            self.answer_output.setText("请输入 URL 和问题。")
            self.references_output.setText("")
            return

        self.status_label.setText("加载中...")
        QApplication.processEvents()

        try:
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
            doc_splits = text_splitter.split_documents(docs_list)
            vector_store = InMemoryVectorStore(embeddings)
            vector_store.add_documents(doc_splits)
        except Exception as e:
            self.status_label.setText("")
            self.answer_output.setText(f"加载文档时出错: {str(e)}")
            self.references_output.setText("")
            return

        self.status_label.setText("")

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

        state = {"messages": [HumanMessage(content=question)]}
        result = self.graph.invoke(state)

        answer = None
        for message in result["messages"]:
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = message.content
                break

        if not answer:
            self.answer_output.setText("未生成回答。")
            self.references_output.setText("")
            return

        # 显示回答
        self.answer_output.setText(answer)




        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )
                
        # LLM with function call
        llm_grader = ChatOpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
            model=open_ai_model,
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

        # Post-processing
        def format_docs(docs):
            return "\n".join(
                f"<doc{i + 1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i + 1}>\n"
                for i, doc in enumerate(docs)
            )

        # 获取检索到的文档并显示参考段落
        retrieved_docs = vector_store.similarity_search(question, k=4)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        docs_to_use = []
        for doc in retrieved_docs:
            res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if res.binary_score == "yes":
                docs_to_use.append(doc)

        lookup_response = doc_lookup.invoke({
            "documents": format_docs(docs_to_use),
            "question": question,
            "generation": answer
        })

        # 格式化参考文档段落
        references_text = ""
        for id, title, source, segment in zip(
            lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment
        ):
            references_text += f"ID: {id}\n标题: {title}\n来源: {source}\n段落: {segment}\n\n"

        self.references_output.setText(references_text if references_text else "未找到相关参考段落。")

# 定义工具和函数
@tool(response_format="content_and_artifact")
def retrieve(question: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(question, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate(state):
    recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"][::-1]
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don’t know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        m for m in state["messages"] if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""
    id: List[str] = Field(..., description="List of id of docs used to answers the question")
    title: List[str] = Field(..., description="List of titles used to answers the question")
    source: List[str] = Field(..., description="List of sources used to answers the question")
    segment: List[str] = Field(..., description="List of direct segments from used documents that answers the question")

parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

Highlightsystem = """You are an advanced assistant for document search and retrieval. You are provided with the following:
1. A question.
2. A generated answer based on the question.
3. A set of documents that were referenced in generating the answer.

Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
in the provided documents.

Ensure that:
- (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
- The relevance of each segment to the generated answer is clear and directly supports the answer provided.
- (Important) If you didn't used the specific document don't mention it.

Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

Highlightprompt = PromptTemplate(
    template=Highlightsystem,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

doc_lookup = Highlightprompt | llm | parser

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
