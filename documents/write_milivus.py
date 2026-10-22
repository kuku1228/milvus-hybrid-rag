import multiprocessing
import os
from multiprocessing import Queue

from LLM_Models.log_utils import log
from documents.milvus_db import MilvusVectorSave
from documents.pdf_parser import PDFParser
from test_load.demo4 import MarkdownParser


# 采用分布式，多进程的方式把海量数据写入Milvus数据库

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下所有md文件和pdf文件并分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    # 获取目录下所有.md和.pdf文件
    all_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.md') or f.endswith('.pdf')
    ]

    if not all_files:
        log.warning("警告：未找到任何.md或.pdf文件")
        output_queue.put(None)  # 发送终止信号
        return

    md_parser = MarkdownParser()
    pdf_parser = PDFParser()
    doc_batch = []
    for file_path in all_files:
        try:
            if file_path.endswith('.md'):
                docs = md_parser.parse_markdown_to_documents(file_path)
            elif file_path.endswith('.pdf'):
                docs = pdf_parser.parse_pdf_to_documents(file_path)
            else:
                continue
            if docs:
                doc_batch.extend(docs)

            # 达到批次大小时发送到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(all_files)}个文件")




def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    log.info("Milvus写入进程启动...")


    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as e:
            log.error(f"写入数据失败 ！")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


if __name__ == '__main__':
    # 配置参数
    data_dir = r'D:\Python\PycharmProjects\RAG\data'  # Markdown文件目录
    queue_maxsize = 20  # 队列最大容量（防止内存溢出）

    mv = MilvusVectorSave()
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(data_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")