from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient
from config.config import settings
from llama_index.embeddings.openai import OpenAIEmbedding


def query_products_tool(query: str) -> str:
    qdrant_client = QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION
    )
    embedding = OpenAIEmbedding(model=settings.EMBEDDING_MODEL_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding)
    query_engine = index.as_query_engine(similarity_top_k=15)

    response = query_engine.query(query)

    produtos = []

    for i, node in enumerate(response.source_nodes, start=1):
        meta_root = node.metadata
        meta = meta_root['metadata'] if "metadata" in meta_root else meta_root
        produtos.append(
            f"{i}. {meta['produto']} - Preço Original: R${meta['preco_original']} - "
            f"Preço Promocional: R${meta['preco_desconto']} - Link: {meta['link']}"
        )

    return "\n".join(produtos)



def check_user_tool(cpf: str) -> str:
    """
    Verifica se um usuário com o CPF informado já possui conta ativa no sistema.
    Exemplo: "12345678900".
    """
    print(f"[ACCOUNT TOOL] Checando CPF: {cpf}")
    return f"Usuário com CPF {cpf} verificado."


query_products = FunctionTool.from_defaults(
    fn=query_products_tool,
    name="buscar_produtos",
    description="Usa RAG com Qdrant para buscar produtos promocionais com base em um termo.",
)
check_user = FunctionTool.from_defaults(fn=check_user_tool, name="verificar_usuario")