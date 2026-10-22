import uuid
import gradio as gr
from langchain.agents import create_agent
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore

from LLM_Models.env_utils import MYSQL_URL
from LLM_Models.my_llm import deepseek_llm
from tools.retriever_tools import search_knowledge_base

SYSTEM_PROMPT = (
    "你是一个专业的大模型技术助手，基于知识库内容回答用户问题。"
    "如果知识库中没有相关信息，请如实告知，不要编造答案。"
)

# 主模型所在节点名：LangChain 1.x 的 create_agent 叫 model，老版 create_react_agent 叫 agent
MODEL_NODES = {"model", "agent"}

CSS = """
.gradio-container { max-width: 920px !important; margin: auto;
  font-family: "PingFang SC","Microsoft YaHei",sans-serif; }
#header { background: linear-gradient(135deg,#4f46e5,#06b6d4); color:#fff;
  padding: 18px 24px; border-radius: 14px; margin-bottom: 14px; }
#header h1 { margin: 0; font-size: 21px; font-weight: 600; }
#header p  { margin: 6px 0 0; font-size: 13px; opacity: .92; }
#chatbot { border-radius: 14px !important; }
footer { display: none !important; }
"""

HEADER = (
    '<div id="header">'
    "<h1>🤖 智识阁 · 大模型技术知识库问答</h1>"
    "<p>混合检索（Dense + Sparse）+ Rerank · 回答大模型 / 深度学习 / Transformer 相关问题</p>"
    "</div>"
)


def _chunk_text(msg) -> str:
    """只从 content 取文本，兼容 str 与 blocks 列表；不碰 .text，避免弃用警告"""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") in (None, "text"):
                parts.append(b.get("text", ""))
        return "".join(parts)
    return ""


def _snapshot(history):
    """每帧新建对象，Gradio 才认得出变化（messages 格式）"""
    return [dict(m) for m in history]


def _stream(agent, question, history, thread_id):
    history = _snapshot(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
    yield _snapshot(history)                       # 先把问题亮出来

    acc = ""
    try:
        for chunk, meta in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="messages",                # 逐 token
        ):
            if meta.get("langgraph_node") not in MODEL_NODES:
                continue                           # 跳过工具/子代理的 LLM 输出

            piece = _chunk_text(chunk)
            if not piece:
                continue

            acc += piece                           # 累加，绝不覆盖
            if acc == history[-1]["content"]:
                continue
            history[-1]["content"] = acc
            yield _snapshot(history)

    except Exception as e:
        history[-1]["content"] = f"⚠️ {type(e).__name__}: {e}"
        yield _snapshot(history)
        return

    if not history[-1]["content"]:
        history[-1]["content"] = "（模型没有返回内容，请检查后端与工具调用日志）"
        yield _snapshot(history)


def build_demo(agent):
    def respond(question, history, thread_id):
        if not question or not question.strip():
            yield history, ""
            return
        for snap in _stream(agent, question.strip(), history, thread_id):
            yield snap, ""

    def reset():
        return [], "", str(uuid.uuid4())           # 换 thread = 后端也真的清空

    with gr.Blocks(title="智识阁 · 知识库问答", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.HTML(HEADER)

        chatbot = gr.Chatbot(elem_id="chatbot", height=540, label="对话")

        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入问题，例如：什么是 Transformer？",
                show_label=False,
                container=False,
                scale=9,
            )
            send = gr.Button("发送", variant="primary", scale=1)

        clear = gr.Button("🗑 清空对话")

        gr.Examples(
            examples=[
                ["什么是 Transformer？"],
                ["解释一下 LoRA 微调的原理"],
                ["RAG 和微调有什么区别？"],
            ],
            inputs=msg,
        )

        thread_state = gr.State(lambda: str(uuid.uuid4()))   # 每个浏览器会话一条线

        msg.submit(respond, [msg, chatbot, thread_state], [chatbot, msg])
        send.click(respond, [msg, chatbot, thread_state], [chatbot, msg])
        clear.click(reset, None, [chatbot, msg, thread_state])

    return demo


if __name__ == "__main__":
    with PyMySQLSaver.from_conn_string(MYSQL_URL) as checkpointer, \
         PyMySQLStore.from_conn_string(MYSQL_URL) as store:
        checkpointer.setup()
        store.setup()

        agent = create_agent(
            model=deepseek_llm,
            tools=[search_knowledge_base],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
            store=store,
        )

        build_demo(agent).launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,          # 出错时把 traceback 打到界面，别再静默
        )