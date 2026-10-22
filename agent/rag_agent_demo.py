from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from LLM_Models.my_llm import deepseek_llm
from tools.retriever_tools import search_knowledge_base

# 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])

# 创建智能体
agent = create_tool_calling_agent(deepseek_llm, [search_knowledge_base], prompt)
executor = AgentExecutor(agent=agent, tools=[search_knowledge_base])


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

resp = agent_with_history.invoke(
    {'input': '什么是神经网络？'},
    config={'configurable': {"session_id": 'zs123'}}
)

print(resp)