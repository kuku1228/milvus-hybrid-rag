import os

from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

MILVUS_URI = os.getenv("MILVUS_URI")

COLLECTION_NAME = 't_collection02'

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

MYSQL_URL = os.getenv("MYSQL_URL")
