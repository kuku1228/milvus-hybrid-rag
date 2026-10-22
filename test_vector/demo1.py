from LLM_Models.embedding_model import bge_embedding

#test
embeddings = bge_embedding.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings), len(embeddings[0]))

embedded_query = bge_embedding.embed_query("What was the name mentioned in the conversation?")
print(embedded_query)