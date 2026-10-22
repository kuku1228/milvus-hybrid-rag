from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

from LLM_Models.embedding_model import bge_embedding
from LLM_Models.env_utils import MILVUS_URI, COLLECTION_NAME
from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave

client = MilvusClient(uri=MILVUS_URI)

def test1():
    # 第一种检索类型：纯密集向量检索（dense）
    # 1. 先用 BGE 把问题转成密集向量
    query_vector = bge_embedding.embed_query("什么是排序算法？")
    # 2. 做密集向量检索
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],  # ← 传入 query 的向量
        anns_field="dense",  # ← 查 dense 字段（密集向量）
        limit=2,  # ← top 2
        filter='category == "content"',  # ← 过滤条件
        search_params={"params": {"ef": 64}},
        output_fields=["text", "title", "category"]
    )
    print("=== 密集向量检索结果 ===")
    for hits in search_result:
        for i, hit in enumerate(hits):
            print(f"\n[{i + 1}] score={hit['distance']:.4f}")
            print(f"    title: {hit['entity'].get('title', '')[:50]}")
            print(f"    text: {hit['entity'].get('text', '')[:100]}...")


def test2():

    # 第二种检索方式：纯 BM25 稀疏检索
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        data=["什么是排序算法？"],  # 你的问题
        anns_field="sparse",  # 查 sparse 字段（BM25 关键词）
        limit=5,  # 返回 top 5
        search_params={
            "params": {"drop_ratio_search": 0.0}
        },
        output_fields=["text", "title", "category"]
    )

    print("=== 纯 BM25 稀疏检索结果 ===")
    for hits in search_result:
        for i, hit in enumerate(hits):
            print(f"\n[{i + 1}] score={hit['distance']:.4f}")
            print(f"    title: {hit['entity'].get('title', '')[:50]}")
            print(f"    text: {hit['entity'].get('text', '')[:100]}...")



def test3():
    # 第三种检索方式： 混合检索
    query = "什么是排序算法？"
    # 密集向量搜索请求
    req1 = AnnSearchRequest(
        data=[bge_embedding.embed_query(query)],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=5
    )

    # 稀疏向量搜索请求
    req2 = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=5
    )

    # 混合检索 + RRF 排序
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2],
        ranker=RRFRanker(60),
        limit=5,
        output_fields=["text", "title", "category"]
    )


    for hits in res:
        for i, item in enumerate(hits):
            print(f"[{i + 1}] score={item['distance']:.4f}")
            print(f"    title: {item['entity'].get('title', '')[:50]}")
            print(f"    text:  {item['entity'].get('text', '')[:100]}...")
            print()




if __name__ == '__main__':
    # test1()  # 测试 密集向量检索
    # test2()  # 测试 稀疏向量检索
    test3()  # 测试 混合检索
