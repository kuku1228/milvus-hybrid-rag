
from langchain.agents import create_agent
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore
from LLM_Models.env_utils import MYSQL_URL, HUGGINGFACE_TOKEN
from LLM_Models.my_llm import deepseek_llm
from tools.retriever_tools import search_knowledge_base




with (PyMySQLSaver.from_conn_string(MYSQL_URL) as checkpointer,
     PyMySQLStore.from_conn_string(MYSQL_URL) as store):
    checkpointer.setup()
    store.setup()

    agent = create_agent(
        model = deepseek_llm,
        tools = [search_knowledge_base],
        system_prompt="你是一个专业的大模型技术助手，基于知识库内容回答用户问题。如果知识库中没有相关信息，请如实告知，不要编造答案。",
        checkpointer=checkpointer,
        store= store
        )


if __name__ == "__main__":
    # ========== 2. 对话循环 ==========

    print("=" * 50)
    print("大模型技术知识库问答系统")
    print("输入问题开始对话，输入 exit 退出")
    print("=" * 50)

    config = {'configurable': {"thread_id": "user_001"}}

    while True:
        question = input("\n请向我提出问题: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "退出"]:
            print("再见！")
            break


        try:
            response = agent.invoke({'messages': [{'role': 'user', 'content': question}]},
                                     config=config)

            print(f"\n助手: {response['messages'][-1].content}")
        except Exception as e:
            print(f"\n出错了: {e}")



