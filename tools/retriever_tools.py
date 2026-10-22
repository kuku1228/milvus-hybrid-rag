from langchain_core.tools import tool

from RAGRetriever.RAGRetriever import RAGRetriever
from config import Top_k_recall, Top_k_rerank

retriever_tool = RAGRetriever(top_k_recall=Top_k_recall, top_k_rerank=Top_k_rerank)

@tool
def search_knowledge_base(query: str) -> str:
    """
    从大模型技术知识库中检索相关文档。
    当用户问深度学习、大模型、神经网络、Transformer、RAG 等技术问题时，使用这个工具查找相关资料。
    Args:
        query: 用户的问题，用自然语言描述
    """
    docs = retriever_tool.retrieve(query)
    if not docs:
        return "未找到相关文档。"

    result = ""
    for i, doc in enumerate(docs):
        result += f"【文档{i + 1}】标题：{doc['title']}\n"
        result += f"内容：{doc['text']}\n\n"
    return result

