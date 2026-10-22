from pymilvus import MilvusClient
from LLM_Models.env_utils import MILVUS_URI, COLLECTION_NAME

client = MilvusClient(uri=MILVUS_URI)

# 1. 总行数
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"总行数: {stats['row_count']}")

# 2. 查前5条看看
result = client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    output_fields=["text", "title", "category"],
    limit=5
)
print("\n=== 前5条数据 ===")
for i, r in enumerate(result):
    print(f"\n[{i+1}] title: {r.get('title', '')[:50]}")
    print(f"    category: {r.get('category', '')}")
    print(f"    text: {r.get('text', '')[:100]}...")

# 3. 测一个 BM25 关键词搜索
search_result = client.search(
    collection_name=COLLECTION_NAME,
    data=["神经网络"],
    anns_field="sparse",
    limit=3,
    search_params={"params": {"drop_ratio_search": 0.0}},
    output_fields=["text", "title"]
)
print("\n=== 搜'神经网络'的结果 ===")
for hits in search_result:
    for hit in hits:
        print(f"  score={hit['distance']:.4f} | {hit['entity']['title'][:40]}")
        print(f"         {hit['entity']['text'][:80]}...")
