
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from LLM_Models.env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, HUGGINGFACE_TOKEN

dashscope_embedding = DashScopeEmbeddings(
    dashscope_api_key=DASHSCOPE_API_KEY,
)


model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}# 归一化向量

bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


