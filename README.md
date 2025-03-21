# 项目名称：RAG智能问答系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)


## 项目概述

该项目为基于 RAG（Retrieval-Augmented Generation） 的智能问答系统，结合检索与生成技术，利用 LangChain 框架实现高效的对话管理功能。

## 技术亮点

#### 应用 RAG 技术，通过检索预加载文档生成准确的回答。
#### 集成 LangChain，实现对话流程管理和上下文记忆。
#### 利用嵌入模型与向量存储，优化语义搜索的高效检索能力。
## 个人贡献
#### 设计并实现了系统架构与核心功能。
#### 优化了检索与生成模块的性能，确保系统运行稳定。
## 项目价值

该项目展示了 RAG 与 LangChain 在智能对话领域的应用潜力，可广泛用于客服支持、知识查询等场景。




## 🛠️ 技术架构

```mermaid
graph TD
    A[用户输入] --> B{混合检索引擎}
    B -->|Dense| C[Faiss向量库]
    B -->|Sparse| D[BM25索引]
    C --> E[相关性重排序]
    D --> E
    E --> F[知识增强生成]
    F --> G[三级验证机制]
    G --> H[结构化输出]
