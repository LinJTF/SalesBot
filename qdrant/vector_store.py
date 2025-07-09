import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from config.config import settings


def setup_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_LINK)


def load_csv_products(csv_path: str) -> list[dict[str, str]]:
    products_raw = pd.read_csv(csv_path).dropna()
    products = products_raw.to_dict(orient="records")
    return products


def generate_openai_embeddings(
    texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [embedding_data.embedding for embedding_data in response.data]


def ensure_collection_exists(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


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


def main() -> None:
    collection_name = "products_promotions"
    csv_path = "docs/rag.csv"
    client = setup_qdrant_client()
    products = load_csv_products(csv_path)
    texts = []
    for product in products:
        text = (
            f"Produto: {product['Produto']}; "
            f"Preço Original: {product['Preço Original']}; "
            f"Preço com Desconto: {product['Preço com Desconto']}; "
            f"Desconto: {product['Desconto']}; "
            f"Link: {product['Link']}"
        )
        texts.append(text)
    embeddings = generate_openai_embeddings(texts)
    ensure_collection_exists(client, collection_name, vector_size=len(embeddings[0]))
    upload_products_to_qdrant(client, collection_name, products, embeddings)
    print(f"Uploaded {len(products)} product vectors successfully.")


if __name__ == "__main__":
    main()
