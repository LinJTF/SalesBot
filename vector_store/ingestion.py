from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def upload_products_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    products: list[dict[str, str]],
    embeddings: list[list[float]],
) -> None:
    points = []
    for idx, (product, embedding) in enumerate(zip(products, embeddings)):
        page_content = (
            f"Produto: {product['Produto']}. "
            f"Preço original: R${product['Preço Original']}. "
            f"Preço com desconto: R${product['Preço com Desconto']} "
            f"({product['Desconto']}). "
            f"Link: {product['Link']}"
        )
        payload = {
            "text": page_content,
            "metadata": {
                "produto": product["Produto"],
                "preco_original": product["Preço Original"],
                "preco_desconto": product["Preço com Desconto"],
                "desconto": product["Desconto"],
                "link": product["Link"],
            },
        }
        points.append(PointStruct(id=idx, vector=embedding, payload=payload))
    client.upsert(collection_name=collection_name, points=points)
