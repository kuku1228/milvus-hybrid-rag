#导入deepseek大模型

from langchain_deepseek import ChatDeepSeek

from LLM_Models.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm = ChatDeepSeek(
    model="deepseek-flash",
    api_key= DEEPSEEK_API_KEY,
    base_url= DEEPSEEK_BASE_URL,
)

