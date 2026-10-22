import dashscope
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from LLM_Models.embedding_model import bge_embedding
from LLM_Models.env_utils import MILVUS_URI, COLLECTION_NAME, DASHSCOPE_API_KEY
from config import Top_k_recall, Top_k_rerank




class RAGRetriever:
    """混合检索 + Rerank 的封装"""

    def __init__(self, top_k_recall=Top_k_recall, top_k_rerank=Top_k_rerank):
        dashscope.api_key = DASHSCOPE_API_KEY
        self.client = MilvusClient(uri=MILVUS_URI)
        self.top_k_recall = top_k_recall    # 召回数量
        self.top_k_rerank = top_k_rerank    # 最终返回数量

    def retrieve(self, query: str):
        """
        输入用户问题，返回 rerank 后的 top_k_rerank 条文档
        每条文档格式: {'text': ..., 'title': ..., 'score': ...}
        """
        # 第一步：混合检索召回
        recall_hits = self._hybrid_recall(query)

        if not recall_hits:
            return []

        # 第二步：Rerank 重排序
        final_docs = self._rerank(query, recall_hits)

        return final_docs

    def _hybrid_recall(self, query: str):
        """dense + sparse 混合召回"""
        req1 = AnnSearchRequest(
            data=[bge_embedding.embed_query(query)],
            anns_field="dense",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=self.top_k_recall
        )

        req2 = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=self.top_k_recall
        )

        results = self.client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[req1, req2],
            ranker=RRFRanker(60),
            limit=self.top_k_recall,
            output_fields=["text", "title", "category"]
        )

        return results[0]

    def _rerank(self, query: str, recall_hits):
        """DashScope GTE-Rerank 重排序"""
        docs_text = [hit['entity']['text'] for hit in recall_hits]

        resp = dashscope.TextReRank.call(
            model='gte-rerank-v2',
            query=query,
            documents=docs_text,
            top_n=self.top_k_rerank,
            return_documents=True
        )


        final_docs = []
        for result in resp.output.results:
            original_hit = recall_hits[result.index]
            final_docs.append({
                'text': original_hit['entity']['text'],
                'title': original_hit['entity'].get('title', ''),
                'category': original_hit['entity'].get('category', ''),
                'rerank_score': result.relevance_score
            })

        return final_docs


